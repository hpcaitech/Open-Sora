import os
from pprint import pformat

import colossalai
import torch
import torch.distributed as dist
from tqdm import tqdm

from opensora.acceleration.parallel_states import get_data_parallel_group, set_data_parallel_group
from opensora.datasets.dataloader import prepare_dataloader
from opensora.registry import DATASETS, MODELS, build_module
from opensora.utils.config_utils import parse_configs, save_training_config
from opensora.utils.misc import FeatureSaver, Timer, create_logger, format_numel_str, get_model_numel, to_torch_dtype


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

    colossalai.launch_from_torch({})
    set_data_parallel_group(dist.group.WORLD)

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
    )
    dataloader, _ = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    num_steps_per_epoch = len(dataloader)

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
    # 5. training loop
    # =======================================================
    # == global variables ==
    bin_size = cfg.bin_size
    save_text_features = cfg.get("save_text_features", False)
    save_compressed_text_features = cfg.get("save_compressed_text_features", False)

    # == number of bins ==
    num_bin = num_steps_per_epoch // bin_size
    logger.info("Number of batches: %s", num_steps_per_epoch)
    logger.info("Bin size: %s", bin_size)
    logger.info("Number of bins: %s", num_bin)

    # resume from a specific batch index
    start_index = cfg.get("start_index", 0)
    end_index = cfg.get("end_index", num_bin)
    dataloader.batch_sampler.load_state_dict({"last_micro_batch_access_index": start_index})
    num_bin_to_process = min(num_bin, end_index) - start_index
    logger.info("Start index: %s", start_index)
    logger.info("End index: %s", end_index)
    logger.info("Number of batches to process: %s", num_bin_to_process)

    # create save directory
    assert cfg.get("save_dir", None) is not None, "Please specify the save_dir in the config file."
    save_dir = os.path.join(cfg.save_dir, f"s{start_index}_e{end_index}")
    os.makedirs(save_dir, exist_ok=True)
    save_training_config(cfg.to_dict(), save_dir)
    logger.info("Saving features to %s", save_dir)

    saver = FeatureSaver(save_dir, bin_size, start_bin=start_index)

    # == training loop in an epoch ==
    dataloader_iter = iter(dataloader)
    log_time = cfg.get("log_time", False)
    for i in tqdm(range(0, num_bin_to_process * bin_size)):
        with Timer("step", log=log_time):
            with Timer("data loading", log=log_time):
                batch = next(dataloader_iter)
                x = batch.pop("video").to(device, dtype)  # [B, C, T, H, W]
                y = batch.pop("text")

            with Timer("vae", log=log_time):
                x = vae.encode(x)
            with Timer("feature to cpu", log=log_time):
                x = x.cpu()

            batch_dict = {
                "x": x,
                "text": y,
                "fps": batch["fps"].to(dtype),
                "height": batch["height"].to(dtype),
                "width": batch["width"].to(dtype),
                "num_frames": batch["num_frames"].to(dtype),
            }

            if save_text_features:
                with Timer("text", log=log_time):
                    text_infos = text_encoder.encode(y)
                    y_feat = text_infos["y"]
                    y_mask = text_infos["mask"]
                    if save_compressed_text_features:
                        y_feat, y_mask = model.encode_text(y_feat, y_mask)
                        y_mask = torch.tensor(y_mask)
                with Timer("feature to cpu", log=log_time):
                    y_feat = y_feat.cpu()
                    y_mask = y_mask.cpu()
                batch_dict.update({"y": y_feat, "mask": y_mask})

            saver.update(batch_dict)


if __name__ == "__main__":
    main()
