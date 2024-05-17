import os
from pprint import pformat

import torch
import torch.distributed as dist
from tqdm import tqdm

from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets import prepare_dataloader, prepare_variable_dataloader
from opensora.datasets.utils import collate_fn_ignore_none
from opensora.registry import DATASETS, MODELS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import FeatureSaver, create_logger, format_numel_str, get_model_numel, to_torch_dtype

DEFAULT_DATASET_NAME = "VideoTextDataset"


def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)

    # == device and dtype ==
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

    # == colossalai init distributed training ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == init logger, tensorboard & wandb ==
    logger = create_logger()
    logger.info("Configuration:\n %s", pformat(cfg.to_dict()))

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")
    # == build dataset ==
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", None),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        collate_fn=collate_fn_ignore_none,
    )
    if cfg.dataset.type == DEFAULT_DATASET_NAME:
        dataloader = prepare_dataloader(**dataloader_args)
        total_batch_size = cfg.batch_size * dist.get_world_size() // cfg.get("sp_size", 1)
        logger.info("Total batch size: %s", total_batch_size)
        num_steps_per_epoch = len(dataloader)
    else:
        dataloader = prepare_variable_dataloader(
            bucket_config=cfg.get("bucket_config", None),
            num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
            **dataloader_args,
        )
        num_steps_per_epoch = dataloader.batch_sampler.get_num_batch()

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device, dtype=dtype)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()

    # == build diffusion model ==
    input_size = (dataset.num_frames, *dataset.image_size)
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
        .train()
    )
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        "[Diffusion] Trainable model params: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel),
    )

    # =======================================================
    # 4. distributed training preparation with colossalai
    # =======================================================
    # == global variables ==
    start_step = sampler_start_idx = 0
    logger.info("Training for with %s steps per epoch", num_steps_per_epoch)

    if cfg.dataset.type == DEFAULT_DATASET_NAME:
        dataloader.sampler.set_start_index(sampler_start_idx)

    # =======================================================
    # 5. training loop
    # =======================================================
    dist.barrier()
    for epoch in range(1):
        # == set dataloader to new epoch ==
        if cfg.dataset.type == DEFAULT_DATASET_NAME:
            dataloader.sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info("Beginning epoch %s...", epoch)

        # == training loop in an epoch ==
        assert cfg.get("save_dir", None) is not None, "Please specify the save_dir in the config file."
        os.makedirs(cfg.save_dir, exist_ok=True)
        saver = FeatureSaver(cfg.save_dir)
        save_text_features = cfg.get("save_text_features", False)
        save_compressed_text_features = cfg.get("save_compressed_text_features", False)

        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            for step, batch in pbar:
                x = batch.pop("video").to(device, dtype)  # [B, C, T, H, W]
                y = batch.pop("text")

                x = vae.encode(x).cpu()  # [B, C, T, H/P, W/P]
                fps = batch["fps"].to(dtype)
                batch_dict = {"x": x, "fps": fps}

                if save_text_features:
                    text_infos = text_encoder.encode(y)
                    y_feat = text_infos["y"]
                    y_mask = text_infos["mask"]
                    if not save_compressed_text_features:
                        y_feat = y_feat.cpu()
                        y_mask = y_mask.cpu()
                    else:
                        y_feat, y_mask = model.encode_text(y_feat, y_mask)
                        y_feat = y_feat.cpu()
                        y_mask = torch.tensor(y_mask)
                        breakpoint()
                    batch_dict.update({"y": y_feat, "mask": y_mask})

                saver.update(batch_dict)


if __name__ == "__main__":
    main()
