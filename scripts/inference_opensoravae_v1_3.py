import json
import os
import resource
from pprint import pformat

import colossalai
import torch
from mmengine.runner import set_random_seed
from tqdm import tqdm

from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.dataloader import prepare_dataloader
from opensora.models.vae_v1_3.losses import VAELoss
from opensora.registry import DATASETS, MODELS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import (
    create_logger,
    format_numel_str,
    get_model_numel,
    get_world_size,
    is_distributed,
    is_main_process,
    to_torch_dtype,
)


def get_memory_usage():
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return rusage.ru_maxrss  # Memory usage in kilobytes


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
    model_numel, model_numel_trainable = get_model_numel(model)
    print("model parameters:", format_numel_str(model_numel))
    max_mem_alloc = torch.cuda.max_memory_allocated(device) / (1024**3)
    max_mem_resv = torch.cuda.max_memory_reserved(device) / (1024**3)
    logger.info("After building CausalVAE model; Max memory allocated: %.4f GiB", max_mem_alloc)
    logger.info("After building CausalVAE model; Max memory reserved: %.4f GiB", max_mem_resv)

    cal_stats = cfg.get("cal_stats", None)
    if cal_stats:
        # == setup loss functions ==

        vae_loss_fn = (
            VAELoss(**cfg.vae_loss_config, device=device, dtype=dtype) if cfg.get("vae_loss_config", None) else None
        )

    # # 显存开销
    # mod_name_mapping = {}
    # for name, mod in model.named_modules():
    #     mod_name_mapping[mod] = name

    # def hook(module, input, output):
    #     print(f"{mod_name_mapping[module]}: {torch.cuda.memory_allocated() / 1024 ** 2} MB")

    # for mod in model.modules():
    #     mod.register_forward_hook(hook)

    # ======================================================
    # inference
    # ======================================================
    # == global variables ==
    running_loss = running_gen_loss = running_disc_loss = 0.0
    loss_info = {}
    if cal_stats:
        num_samples = 0
        running_sum = running_var = 0.0
        running_sum_c_head = torch.zeros(model.out_channels, dtype=torch.float, device=device)
        running_var_c_head = torch.zeros(model.out_channels, dtype=torch.float, device=device)
        running_sum_c_tail = torch.zeros(model.out_channels, dtype=torch.float, device=device)
        running_var_c_tail = torch.zeros(model.out_channels, dtype=torch.float, device=device)

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
            z, x_rec, posterior = model(x, is_training=False)

            # == save samples ==
            save_dir = cfg.get("save_dir", None)
            if is_main_process() and save_dir is not None:
                ori_dir = f"{save_dir}_ori"
                rec_dir = f"{save_dir}_rec"
                os.makedirs(ori_dir, exist_ok=True)
                os.makedirs(rec_dir, exist_ok=True)
                for idx, vid in enumerate(x):
                    pos = step * cfg.batch_size + idx
                    save_sample(vid, fps=save_fps, save_path=f"{ori_dir}/{pos:03d}", verbose=verbose >= 2)
                    save_sample(x_rec[idx], fps=save_fps, save_path=f"{rec_dir}/{pos:03d}", verbose=verbose >= 2)

            if cal_stats:
                num_samples += 1
                running_sum += z.mean().item()
                running_var += (z - running_sum / num_samples).pow(2).mean().item()

                running_sum_c += z.mean(dim=(0, 2, 3, 4)).float()
                running_var_c += (
                    (z - running_sum_c_head[None, :, None, None, None] / num_samples)
                    .pow(2)
                    .mean(dim=(0, 2, 3, 4))
                    .float()
                )

                running_sum_c_head += z[:, :, :1, :, :].mean(dim=(0, 2, 3, 4)).float()
                running_var_c_head += (
                    (z[:, :, :1, :, :] - running_sum_c_head[None, :, None, None, None] / num_samples)
                    .pow(2)
                    .mean(dim=(0, 2, 3, 4))
                    .float()
                )

                running_sum_c_tail += z[:, :, 1:, :, :].mean(dim=(0, 2, 3, 4)).float()
                running_var_c_tail += (
                    (z[:, :, 1:, :, :] - running_sum_c_tail[None, :, None, None, None] / num_samples)
                    .pow(2)
                    .mean(dim=(0, 2, 3, 4))
                    .float()
                )

                if vae_loss_fn:
                    # == reconstruction loss ==
                    nll_loss, weighted_nll_loss, weighted_kl_loss = vae_loss_fn(x, x_rec, posterior)
                    vae_loss = weighted_kl_loss + weighted_nll_loss

                    running_loss = vae_loss.item() / num_samples + running_loss * ((num_samples - 1) / num_samples)

                if verbose >= 1:
                    pbar.set_postfix(
                        {
                            "mean": running_sum / num_samples,
                            "std": (running_var / num_samples) ** 0.5,
                        }
                    )
                if num_samples % cfg.get("log_stats_every", 100) == 0:
                    logger.info(
                        "[OVERALL] VAE feature per channel stats: mean %s, std %s",
                        (running_sum_c / num_samples).cpu().tolist(),
                        (running_var_c / num_samples).sqrt().cpu().tolist(),
                    )

                    logger.info(
                        "[HEAD] VAE feature per channel stats: mean %s, std %s",
                        (running_sum_c_head / num_samples).cpu().tolist(),
                        (running_var_c_head / num_samples).sqrt().cpu().tolist(),
                    )
                    logger.info(
                        "[TAIL] VAE feature per channel stats: mean %s, std %s",
                        (running_sum_c_tail / num_samples).cpu().tolist(),
                        (running_var_c_tail / num_samples).sqrt().cpu().tolist(),
                    )

    if cal_stats:
        logger.info("total vae loss: %s", running_loss)
        loss_info["total_vae_loss"] = running_loss
        if cfg.model.from_pretrained:
            if os.path.isdir(cfg.model.from_pretrained):
                loss_dir = os.path.join(cfg.model.from_pretrained, "eval")
            else:
                loss_dir = os.path.join(os.path.dirname(cfg.model.from_pretrained), "eval")
            os.makedirs(loss_dir, exist_ok=True)
            output_file_path = os.path.join(
                loss_dir, "loss_" + str(cfg.num_frames) + "f_" + str(cfg.image_size[0]) + "res.json"
            )
            with open(output_file_path, "w") as outfile:
                json.dump(loss_info, outfile, indent=4, sort_keys=True)
            print(f"results saved to: {output_file_path}")

    max_mem_alloc = torch.cuda.max_memory_allocated(device) / (1024**3)
    max_mem_resv = torch.cuda.max_memory_reserved(device) / (1024**3)
    logger.info("Max memory allocated: %.4f GiB", max_mem_alloc)
    logger.info("Max memory reserved: %.4f GiB", max_mem_resv)


if __name__ == "__main__":
    main()
