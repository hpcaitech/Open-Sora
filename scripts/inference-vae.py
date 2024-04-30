import os

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm

from opensora.acceleration.parallel_states import get_data_parallel_group, set_sequence_parallel_group
from opensora.datasets import prepare_dataloader, save_sample
from opensora.models.vae.losses import VAELoss
from opensora.registry import DATASETS, MODELS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype


def main():
    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    cfg = parse_configs(training=False)
    print(cfg)

    # init distributed
    if os.environ.get("WORLD_SIZE", None):
        use_dist = True
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()

        if coordinator.world_size > 1:
            set_sequence_parallel_group(dist.group.WORLD)
        else:
            pass
    else:
        use_dist = False

    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.dtype)
    set_random_seed(seed=cfg.seed)

    # ======================================================
    # 3. build dataset and dataloader
    # ======================================================
    dataset = build_module(cfg.dataset, DATASETS)
    dataloader = prepare_dataloader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )
    print(f"Dataset contains {len(dataset):,} videos ({cfg.dataset.data_path})")
    total_batch_size = cfg.batch_size * dist.get_world_size() // cfg.sp_size
    print(f"Total batch size: {total_batch_size}")

    # ======================================================
    # 4. build model & load weights
    # ======================================================
    # 3.1. build model
    if cfg.get("vae_2d", None) is not None:
        vae_2d = build_module(cfg.vae_2d, MODELS)
        vae_2d.to(device, dtype).eval()
    model = build_module(
        cfg.model,
        MODELS,
        device=device,
        dtype=dtype,
    )
    # discriminator = build_module(cfg.discriminator, MODELS, device=device)

    # 3.2. move to device & eval
    # discriminator = discriminator.to(device, dtype).eval()

    # 3.4. support for multi-resolution
    # model_args = dict()
    # if cfg.multi_resolution:
    #     image_size = cfg.dataset.image_size
    #     hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
    #     ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
    #     model_args["data_info"] = dict(ar=ar, hw=hw)

    # ======================================================
    # 4. inference
    # ======================================================
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 4.1. batch generation

    # define loss function
    if cfg.calc_loss:
        vae_loss_fn = VAELoss(
            logvar_init=cfg.logvar_init,
            perceptual_loss_weight=cfg.perceptual_loss_weight,
            kl_loss_weight=cfg.kl_loss_weight,
            device=device,
            dtype=dtype,
        )

        # adversarial_loss_fn = AdversarialLoss(
        #     discriminator_factor=cfg.discriminator_factor,
        #     discriminator_start=cfg.discriminator_start,
        #     generator_factor=cfg.generator_factor,
        #     generator_loss_type=cfg.generator_loss_type,
        # )

        # disc_loss_fn = DiscriminatorLoss(
        #     discriminator_factor=cfg.discriminator_factor,
        #     discriminator_start=cfg.discriminator_start,
        #     discriminator_loss_type=cfg.discriminator_loss_type,
        #     lecam_loss_weight=cfg.lecam_loss_weight,
        #     gradient_penalty_loss_weight=cfg.gradient_penalty_loss_weight,
        # )

        # # LeCam EMA for discriminator

        # lecam_ema = LeCamEMA(decay=cfg.ema_decay, dtype=dtype, device=device)

    running_loss = 0.0
    running_nll = 0.0
    loss_steps = 0

    # disc_time_downsample_factor = 2 ** len(cfg.discriminator.channel_multipliers)
    # if cfg.dataset.num_frames % disc_time_downsample_factor != 0:
    #     disc_time_padding = disc_time_downsample_factor - cfg.dataset.num_frames % disc_time_downsample_factor
    # else:
    #     disc_time_padding = 0

    total_steps = len(dataloader)
    if cfg.max_test_samples is not None:
        total_steps = min(int(cfg.max_test_samples // cfg.batch_size), total_steps)
        print(f"limiting test dataset to {int(cfg.max_test_samples//cfg.batch_size) * cfg.batch_size}")
    dataloader_iter = iter(dataloader)

    with tqdm(
        range(total_steps),
        disable=not coordinator.is_master(),
        total=total_steps,
        initial=0,
    ) as pbar:
        for step in pbar:
            batch = next(dataloader_iter)
            x = batch["video"].to(device, dtype)  # [B, C, T, H, W]

            #  ===== Spatial VAE =====
            if cfg.get("vae_2d", None) is not None:
                x_z = vae_2d.encode(x)
                x_z_debug = vae_2d.decode(x_z)

            #  ====== VAE ======
            x_z_rec, posterior, z = model(x_z)
            x_rec = vae_2d.decode(x_z_rec)

            if cfg.calc_loss:
                # simple nll loss
                nll_loss, weighted_nll_loss, weighted_kl_loss = vae_loss_fn(x, x_rec, posterior)

                # fake_video = pad_at_dim(recon_video, (disc_time_padding, 0), value=0.0, dim=2)
                # fake_logits = discriminator(fake_video.contiguous())
                # adversarial_loss = adversarial_loss_fn(
                #     fake_logits,
                #     nll_loss,
                #     vae.get_last_layer(),
                #     cfg.discriminator_start + 1,  # Hack to use discriminator
                #     is_training=vae.training,
                # )

                # vae_loss = weighted_nll_loss + weighted_kl_loss + adversarial_loss
                vae_loss = weighted_nll_loss + weighted_kl_loss

                # #  ====== Discriminator Loss ======
                # real_video = pad_at_dim(video, (disc_time_padding, 0), value=0.0, dim=2)
                # fake_video = pad_at_dim(recon_video, (disc_time_padding, 0), value=0.0, dim=2)

                # if cfg.gradient_penalty_loss_weight is not None and cfg.gradient_penalty_loss_weight > 0.0:
                #     real_video = real_video.requires_grad_()
                #     real_logits = discriminator(
                #         real_video.contiguous()
                #     )  # SCH: not detached for now for gradient_penalty calculation
                # else:
                #     real_logits = discriminator(real_video.contiguous().detach())

                # fake_logits = discriminator(fake_video.contiguous().detach())

                # lecam_ema_real, lecam_ema_fake = lecam_ema.get()
                # weighted_d_adversarial_loss, lecam_loss, gradient_penalty_loss = disc_loss_fn(
                #     real_logits,
                #     fake_logits,
                #     cfg.discriminator_start + 1,  # Hack to use discriminator
                #     lecam_ema_real=lecam_ema_real,
                #     lecam_ema_fake=lecam_ema_fake,
                #     real_video=real_video if cfg.gradient_penalty_loss_weight is not None else None,
                # )

                # disc_loss = weighted_d_adversarial_loss + lecam_loss + gradient_penalty_loss

                loss_steps += 1
                # running_disc_loss = disc_loss.item() / loss_steps + running_disc_loss * ((loss_steps - 1) / loss_steps)
                running_loss = vae_loss.item() / loss_steps + running_loss * ((loss_steps - 1) / loss_steps)
                running_nll = nll_loss.item() / loss_steps + running_nll * ((loss_steps - 1) / loss_steps)

            #  ===== Spatial VAE =====

            if not use_dist or coordinator.is_master():
                for idx in range(len(x)):
                    pos = step * cfg.batch_size + idx
                    save_path = os.path.join(save_dir, f"sample_{pos}")
                    save_sample(x[idx], fps=cfg.fps, save_path=save_path + "_original")
                    save_sample(x_rec[idx], fps=cfg.fps, save_path=save_path + "_pipeline")
                    if cfg.get("vae_2d", None) is not None:
                        save_sample(x_z_debug[idx], fps=cfg.fps, save_path=save_path + "_2d")

                # if cfg.get("use_pipeline") == True:
                #     for idx, (sample_original, sample_pipeline, sample_2d) in enumerate(
                #         zip(video, recon_video, recon_2d)
                #     ):
                #         pos = step * cfg.batch_size + idx
                #         save_path = os.path.join(save_dir, f"sample_{pos}")
                #         save_sample(sample_original, fps=cfg.fps, save_path=save_path + "_original")
                #         save_sample(sample_2d, fps=cfg.fps, save_path=save_path + "_2d")
                #         save_sample(sample_pipeline, fps=cfg.fps, save_path=save_path + "_pipeline")

                # else:
                #     for idx, (original, recon) in enumerate(zip(video, recon_video)):
                #         pos = step * cfg.batch_size + idx
                #         save_path = os.path.join(save_dir, f"sample_{pos}")
                #         save_sample(original, fps=cfg.fps, save_path=save_path + "_original")
                #         save_sample(recon, fps=cfg.fps, save_path=save_path + "_recon")

    if cfg.calc_loss:
        print("test vae loss:", running_loss)
        print("test nll loss:", running_nll)
        # print("test disc loss:", running_disc_loss)


if __name__ == "__main__":
    main()
