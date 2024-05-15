import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from copy import deepcopy
from datetime import timedelta
from pprint import pformat

import torch
import numpy as np
from tqdm import tqdm
from mmengine.runner import set_random_seed
from torch.utils.data import DataLoader as Dataloader
import random
import json

from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets import prepare_dataloader, prepare_variable_dataloader
from opensora.datasets.aspect import *
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import load, model_gathering, model_sharding, record_model_param_shape, save
from opensora.utils.config_utils import (
    create_tensorboard_writer,
    define_experiment_workspace,
    parse_configs,
    save_training_config,
)
from opensora.utils.misc import (
    all_reduce_mean,
    create_logger,
    format_numel_str,
    get_model_numel,
    requires_grad,
    to_torch_dtype,
)
from opensora.utils.train_utils import MaskGenerator, create_colossalai_plugin, update_ema

DEFAULT_DATASET_NAME = "VideoTextDataset"


@torch.no_grad()
def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==

    cfg = parse_configs(training=True)
    device = torch.device("cuda")

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    set_random_seed(seed=cfg.seed)

    # == init exp_dir ==
    exp_name, exp_dir = define_experiment_workspace(cfg)
    os.makedirs(exp_dir, exist_ok=False)
    

    # == init logger, build public models ==
    # logger = create_logger(exp_dir)

    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype)
    vae.eval()
    model = build_module(
        cfg.model,
        MODELS,
        input_size=(None, None, None),
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
    ).to(device, dtype)
    model.eval()

    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    if cfg.get("mask_ratios", None) is not None:
        mask_generator = MaskGenerator(cfg.mask_ratios)

    torch.set_default_dtype(dtype)
    torch.set_default_dtype(torch.float)

    # start evaluation, prepare a dataset everytime in the loop
    eval_config = cfg.get("eval_config")
    assert eval_config is not None, "eval_config is required for evaluation"

    evaluation_losses = {}
    for res, v in eval_config.items():
        loss_res = {}
        for num_frames, (_,batch_size) in v.items():
            loss_frame = {}
            # for each resolution, there may be different aspect ratios(image_size), can be found in datasets/aspect.py
            with tqdm(ASPECT_RATIOS[res][1].items(), desc=f"Resolution {res} num_frames {num_frames}") as pbar:
                for ar, img_size in pbar:
                    # == build dataset ==
                    dataset = build_module({
                        "type": DEFAULT_DATASET_NAME,
                        "num_frames": num_frames,
                        "data_path": cfg.dataset['data_path'],
                        "frame_interval": 1,
                        "image_size": img_size,
                        "transform_name":"resize_crop",
                    }, DATASETS)
                    # == build dataloader ==
                    seed=cfg.get("seed", 1024)
                    def seed_worker(worker_id):
                        worker_seed = seed
                        np.random.seed(worker_seed)
                        torch.manual_seed(worker_seed)
                        random.seed(worker_seed)
                    dataloader_args = dict(
                        dataset=dataset,
                        batch_size=batch_size,
                        num_workers=cfg.get("num_workers", 4),
                        # seed=cfg.get("seed", 1024),
                        shuffle=True,
                        drop_last=True,
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                    )
                    dataloader = Dataloader(
                        **dataloader_args,
                    )
                    # dataloader.sampler.set_start_index(0)
                    dataloader_iter = iter(dataloader)
                    num_steps_per_t = cfg.num_eval_samples // batch_size
                    loss_ar = {}

                    for t in range(0, scheduler.num_timesteps, scheduler.num_timesteps//cfg.num_eval_timesteps):
                        # save key = (res, num_frames, ar, t), value = loss finally
                        loss_t = None
                        for estep in range(num_steps_per_t):
                            batch = next(dataloader_iter)
                            x = batch.pop("video").to(device, dtype)
                            y = batch.pop("text")
                            x = vae.encode(x)
                            model_args = text_encoder.encode(y)
                            model_args["x_mask"] = None

                            mask = None
                            if cfg.get("mask_ratios", None) is not None:
                                mask = mask_generator.get_masks(x)
                                model_args["x_mask"] = mask

                            # == video meta info ==
                            for k, v in batch.items():
                                model_args[k] = v.to(device, dtype)

                            
                            # add height, width and num_frame since they are not in batch meta info
                            model_args["height"] = torch.tensor([img_size[0]], device=device, dtype=dtype)
                            model_args["width"] = torch.tensor([img_size[1]], device=device, dtype=dtype)
                            model_args['num_frames'] = torch.tensor([num_frames], device=device, dtype=dtype)

                            # == diffusion loss computation ==
                            timestep = torch.tensor([t]*x.shape[0], device=device, dtype=dtype)
                            loss_dict = scheduler.training_losses(model, x, model_args, mask=mask, t=timestep)
                            losses = loss_dict["loss"] # (batch_size)
                            loss_t = losses if loss_t is None else torch.cat([loss_t, losses], dim=0)
                        # save the avg loss for this tuple(res, num_frames, ar, t)
                        loss_ar[t] = loss_t.mean().item(), loss_t.std().item()
                        pbar.set_postfix({"ar": ar, "t": t, "loss": loss_ar[t][0]})
                    loss_frame[ar] = loss_ar
                    
            loss_res[num_frames] = loss_frame
        evaluation_losses[res] = loss_res
        with open(os.path.join(exp_dir, "evaluation_losses.json"), "w") as f:
            json.dump(evaluation_losses, f)

    # save the evaluation_losses
    

if __name__ == "__main__":
    main()
