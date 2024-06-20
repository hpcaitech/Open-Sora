from pprint import pformat

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm

from opensora.acceleration.parallel_states import get_data_parallel_group, set_data_parallel_group
from opensora.datasets.dataloader import prepare_dataloader
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import create_logger, to_torch_dtype
from opensora.utils.train_utils import MaskGenerator


def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == init distributed env ==
    colossalai.launch_from_torch({})
    DistCoordinator()
    set_random_seed(seed=cfg.get("seed", 1024))
    set_data_parallel_group(dist.group.WORLD)

    # == init logger ==
    logger = create_logger()
    logger.info("Eval loss configuration:\n %s", pformat(cfg.to_dict()))

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()

    # == build diffusion model ==
    input_size = (None, None, None)
    latent_size = vae.get_latent_size(input_size)
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
        )
        .to(device, dtype)
        .eval()
    )
    text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance

    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    if cfg.get("mask_ratios", None) is not None:
        mask_generator = MaskGenerator(cfg.mask_ratios)

    # ======================================================
    # inference
    # ======================================================
    # start evaluation, prepare a dataset everytime in the loop
    bucket_config = cfg.bucket_config
    if cfg.get("resolution", None) is not None:
        bucket_config = {cfg.resolution: bucket_config[cfg.resolution]}
    assert bucket_config is not None, "bucket_config is required for evaluation"
    logger.info("Evaluating bucket_config: %s", bucket_config)

    def build_dataset(resolution, num_frames, batch_size):
        bucket_config = {resolution: {num_frames: (1.0, batch_size)}}
        dataset = build_module(cfg.dataset, DATASETS)
        dataloader_args = dict(
            dataset=dataset,
            batch_size=None,
            num_workers=cfg.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            process_group=get_data_parallel_group(),
        )
        dataloader, sampler = prepare_dataloader(bucket_config=bucket_config, **dataloader_args)
        num_batch = sampler.get_num_batch()
        num_steps_per_epoch = num_batch // dist.get_world_size()
        return dataloader, num_steps_per_epoch, num_batch

    evaluation_losses = {}
    start = cfg.start_index if "start_index" in cfg else 0
    end = cfg.end_index if "end_index" in cfg else len(bucket_config)
    for i, res in enumerate(bucket_config):
        if i < start or i >= end:  # skip task
            continue

        t_bucket = bucket_config[res]
        for num_frames, (_, batch_size) in t_bucket.items():
            if batch_size is None:
                continue
            logger.info("Evaluating resolution: %s, num_frames: %s", res, num_frames)
            dataloader, num_steps_per_epoch, num_batch = build_dataset(res, num_frames, batch_size)
            if num_batch == 0:
                logger.warning("No data for resolution: %s, num_frames: %s", res, num_frames)
                continue

            evaluation_t_losses = []
            for t in torch.linspace(0, scheduler.num_timesteps, cfg.get("num_eval_timesteps", 10) + 2)[1:-1]:
                loss_t = 0.0
                num_samples = 0
                dataloader_iter = iter(dataloader)
                for _ in tqdm(range(num_steps_per_epoch), desc=f"res: {res}, num_frames: {num_frames}, t: {t:.2f}"):
                    batch = next(dataloader_iter)
                    x = batch.pop("video").to(device, dtype)
                    y = batch.pop("text")
                    x = vae.encode(x)
                    model_args = text_encoder.encode(y)

                    # == mask ==
                    mask = None
                    if cfg.get("mask_ratios", None) is not None:
                        mask = mask_generator.get_masks(x)
                        model_args["x_mask"] = mask

                    # == video meta info ==
                    for k, v in batch.items():
                        model_args[k] = v.to(device, dtype)

                    # == diffusion loss computation ==
                    timestep = torch.tensor([t] * x.shape[0], device=device, dtype=dtype)
                    loss_dict = scheduler.training_losses(model, x, model_args, mask=mask, t=timestep)
                    losses = loss_dict["loss"]  # (batch_size)
                    num_samples += x.shape[0]
                    loss_t += losses.sum().item()
                loss_t /= num_samples
                evaluation_t_losses.append(loss_t)
                logger.info("resolution: %s, num_frames: %s, timestep: %.2f, loss: %.4f", res, num_frames, t, loss_t)

            evaluation_losses[(res, num_frames)] = sum(evaluation_t_losses) / len(evaluation_t_losses)
            logger.info(
                "Evaluation losses for resolution: %s, num_frames: %s, loss: %s\n %s",
                res,
                num_frames,
                evaluation_losses[(res, num_frames)],
                evaluation_t_losses,
            )
    logger.info("Evaluation losses: %s", evaluation_losses)


if __name__ == "__main__":
    main()
