import os
from pprint import pformat

import colossalai
import torch
from mmengine.runner import set_random_seed
from tqdm import tqdm

from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.dataloader import prepare_dataloader
from opensora.models.vae.losses import VAELoss
from opensora.registry import DATASETS, MODELS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import create_logger, get_world_size, is_distributed, is_main_process, to_torch_dtype


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
    if is_distributed():
        colossalai.launch_from_torch({})
    set_random_seed(seed=cfg.get("seed", 1024))

    # == init logger ==
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)

    # ======================================================
    # build dataset and dataloader
    # ======================================================
    logger.info("Building reconstruction dataset...")
    dataset = build_module(cfg.dataset, DATASETS)
    batch_size = cfg.get("batch_size", 1)
    dataloader, _ = prepare_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg.get("num_workers", 4),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )
    logger.info("Dataset %s contains %s videos.", cfg.dataset.data_path, len(dataset))
    total_batch_size = batch_size * get_world_size()
    logger.info("Total batch size: %s", total_batch_size)

    total_steps = len(dataloader)
    if cfg.get("num_samples", None) is not None:
        total_steps = min(int(cfg.num_samples // cfg.batch_size), total_steps)
        logger.info("limiting test dataset to %s", int(cfg.num_samples // cfg.batch_size) * cfg.batch_size)
    dataiter = iter(dataloader)

    # ======================================================
    # build model & loss
    # ======================================================
    logger.info("Building models...")
    model = build_module(cfg.model, MODELS).to(device, dtype).eval()
    vae_loss_fn = VAELoss(
        logvar_init=cfg.get("logvar_init", 0.0),
        perceptual_loss_weight=cfg.get("perceptual_loss_weight", 0.1),
        kl_loss_weight=cfg.get("kl_loss_weight", 1e-6),
        device=device,
        dtype=dtype,
    )

    # ======================================================
    # inference
    # ======================================================
    # == global variables ==
    running_loss = running_nll = running_nll_z = 0.0
    loss_steps = 0
    cal_stats = cfg.get("cal_stats", False)
    if cal_stats:
        num_samples = 0
        running_sum = running_var = 0.0
        running_sum_c = torch.zeros(model.out_channels, dtype=torch.float, device=device)
        running_var_c = torch.zeros(model.out_channels, dtype=torch.float, device=device)

    # prepare arguments
    save_fps = cfg.get("fps", 24) // cfg.get("frame_interval", 1)

    # Iter over the dataset
    with tqdm(
        range(total_steps),
        disable=not is_main_process() or verbose < 1,
        total=total_steps,
        initial=0,
    ) as pbar:
        for step in pbar:
            batch = next(dataiter)
            x = batch["video"].to(device, dtype)  # [B, C, T, H, W]

            # == vae encoding & decoding ===
            z, posterior, x_z = model.encode(x)
            x_rec, x_z_rec = model.decode(z, num_frames=x.size(2))
            x_ref = model.spatial_vae.decode(x_z)

            # == check z shape ==
            input_size = x.shape[2:]
            latent_size = model.get_latent_size(input_size)
            assert list(z.shape[2:]) == latent_size, f"z shape: {z.shape}, latent_size: {latent_size}"

            # == calculate stats ==
            if cal_stats:
                num_samples += 1
                running_sum += z.mean().item()
                running_var += (z - running_sum / num_samples).pow(2).mean().item()
                running_sum_c += z.mean(dim=(0, 2, 3, 4)).float()
                running_var_c += (
                    (z - running_sum_c[None, :, None, None, None] / num_samples).pow(2).mean(dim=(0, 2, 3, 4)).float()
                )
                if verbose >= 1:
                    pbar.set_postfix(
                        {
                            "mean": running_sum / num_samples,
                            "std": (running_var / num_samples) ** 0.5,
                        }
                    )
                if num_samples % cfg.get("log_stats_every", 100) == 0:
                    logger.info(
                        "VAE feature per channel stats: mean %s, var %s",
                        (running_sum_c / num_samples).cpu().tolist(),
                        (running_var_c / num_samples).sqrt().cpu().tolist(),
                    )

            # == loss calculation ==
            nll_loss, weighted_nll_loss, weighted_kl_loss = vae_loss_fn(x, x_rec, posterior)
            nll_loss_z, _, _ = vae_loss_fn(x_z, x_z_rec, posterior, no_perceptual=True)
            vae_loss = weighted_nll_loss + weighted_kl_loss
            loss_steps += 1
            running_loss = vae_loss.item() / loss_steps + running_loss * ((loss_steps - 1) / loss_steps)
            running_nll = nll_loss.item() / loss_steps + running_nll * ((loss_steps - 1) / loss_steps)
            running_nll_z = nll_loss_z.item() / loss_steps + running_nll_z * ((loss_steps - 1) / loss_steps)

            # == save samples ==
            save_dir = cfg.get("save_dir", None)
            if is_main_process() and save_dir is not None:
                ori_dir = f"{save_dir}_ori"
                rec_dir = f"{save_dir}_rec"
                ref_dir = f"{save_dir}_spatial"
                os.makedirs(ori_dir, exist_ok=True)
                os.makedirs(rec_dir, exist_ok=True)
                os.makedirs(ref_dir, exist_ok=True)
                for idx, vid in enumerate(x):
                    pos = step * cfg.batch_size + idx
                    save_sample(vid, fps=save_fps, save_path=f"{ori_dir}/{pos:03d}", verbose=verbose >= 2)
                    save_sample(x_rec[idx], fps=save_fps, save_path=f"{rec_dir}/{pos:03d}", verbose=verbose >= 2)
                    save_sample(x_ref[idx], fps=save_fps, save_path=f"{ref_dir}/{pos:03d}", verbose=verbose >= 2)

    logger.info("VAE loss: %s", running_loss)
    logger.info("VAE nll loss: %s", running_nll)
    logger.info("VAE nll_z loss: %s", running_nll_z)


if __name__ == "__main__":
    main()
