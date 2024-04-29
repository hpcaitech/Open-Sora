import os
from datetime import timedelta
from pprint import pprint

import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from tqdm import tqdm

import wandb
from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)
from opensora.acceleration.plugin import ZeroSeqParallelPlugin
from opensora.datasets import prepare_dataloader
from opensora.models.vae.losses import AdversarialLoss, DiscriminatorLoss, VAELoss
from opensora.registry import DATASETS, MODELS, build_module
from opensora.utils.ckpt_utils import create_logger, load_json, save_json
from opensora.utils.config_utils import (
    create_experiment_workspace,
    create_tensorboard_writer,
    parse_configs,
    save_training_config,
)
from opensora.utils.misc import all_reduce_mean, format_numel_str, get_model_numel, to_torch_dtype


def main():
    # ======================================================
    # 1. args & cfg
    # ======================================================
    cfg = parse_configs(training=True)
    exp_name, exp_dir = create_experiment_workspace(cfg)
    save_training_config(cfg._cfg_dict, exp_dir)

    # ======================================================
    # 2. runtime variables & colossalai launch
    # ======================================================
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert cfg.dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg.dtype}"

    # 2.1. colossalai init distributed training
    # we set a very large timeout to avoid some processes exit early
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(1024)
    coordinator = DistCoordinator()
    device = get_current_device()
    dtype = to_torch_dtype(cfg.dtype)

    # 2.2. init logger, tensorboard & wandb
    if not coordinator.is_master():
        logger = create_logger(None)
    else:
        print("Training configuration:")
        pprint(cfg._cfg_dict)
        logger = create_logger(exp_dir)
        logger.info(f"Experiment directory created at {exp_dir}")

        writer = create_tensorboard_writer(exp_dir)
        if cfg.wandb:
            wandb.init(project="minisora", name=exp_name, config=cfg._cfg_dict)

    # 2.3. initialize ColossalAI booster
    if cfg.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=cfg.dtype,
            initial_scale=2**16,
            max_norm=cfg.grad_clip,
        )
        set_data_parallel_group(dist.group.WORLD)
    elif cfg.plugin == "zero2-seq":
        plugin = ZeroSeqParallelPlugin(
            sp_size=cfg.sp_size,
            stage=2,
            precision=cfg.dtype,
            initial_scale=2**16,
            max_norm=cfg.grad_clip,
        )
        set_sequence_parallel_group(plugin.sp_group)
        set_data_parallel_group(plugin.dp_group)
    else:
        raise ValueError(f"Unknown plugin {cfg.plugin}")
    booster = Booster(plugin=plugin)

    # ======================================================
    # 3. build dataset and dataloader
    # ======================================================
    assert cfg.dataset.type == "VideoTextDataset", "Only support VideoTextDataset for now"
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info(f"Dataset contains {len(dataset)} samples.")
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )
    # TODO: use plugin's prepare dataloader
    dataloader = prepare_dataloader(**dataloader_args)
    total_batch_size = cfg.batch_size * dist.get_world_size() // cfg.sp_size
    logger.info(f"Total batch size: {total_batch_size}")

    # ======================================================
    # 4. build model
    # ======================================================
    # 4.1. build model
    if cfg.get("vae_2d", None) is not None:
        vae_2d = build_module(cfg.vae_2d, MODELS)
        vae_2d.to(device, dtype).eval()

    model = build_module(
        cfg.model,
        MODELS,
        device=device,
        dtype=dtype,
    )
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        f"Trainable model params: {format_numel_str(model_numel_trainable)}, Total model params: {format_numel_str(model_numel)}"
    )

    # discriminator = build_module(cfg.discriminator, MODELS, device=device)
    # discriminator_numel, discriminator_numel_trainable = get_model_numel(discriminator)
    # logger.info(
    #     f"Trainable discriminator params: {format_numel_str(discriminator_numel_trainable)}, Total model params: {format_numel_str(discriminator_numel)}"
    # )

    # # LeCam Initialization
    # lecam_ema = LeCamEMA(decay=cfg.ema_decay, dtype=dtype, device=device)

    # 4.3. move to device
    model = model.to(device, dtype)
    # discriminator = discriminator.to(device, dtype)

    # 4.4 loss functions
    vae_loss_fn = VAELoss(
        logvar_init=cfg.logvar_init,
        perceptual_loss_weight=cfg.perceptual_loss_weight,
        kl_loss_weight=cfg.kl_loss_weight,
        device=device,
        dtype=dtype,
    )

    adversarial_loss_fn = AdversarialLoss(
        discriminator_factor=cfg.discriminator_factor,
        discriminator_start=cfg.discriminator_start,
        generator_factor=cfg.generator_factor,
        generator_loss_type=cfg.generator_loss_type,
    )

    disc_loss_fn = DiscriminatorLoss(
        discriminator_factor=cfg.discriminator_factor,
        discriminator_start=cfg.discriminator_start,
        discriminator_loss_type=cfg.discriminator_loss_type,
        lecam_loss_weight=cfg.lecam_loss_weight,
        gradient_penalty_loss_weight=cfg.gradient_penalty_loss_weight,
    )

    # 4.5. setup optimizer
    # vae optimizer
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=0, adamw_mode=True
    )
    lr_scheduler = None
    # disc optimizer
    # disc_optimizer = HybridAdam(
    #     filter(lambda p: p.requires_grad, discriminator.parameters()), lr=cfg.lr, weight_decay=0, adamw_mode=True
    # )
    # disc_lr_scheduler = None

    # 4.6. prepare for training
    if cfg.grad_checkpoint:
        set_grad_checkpoint(model)
        # set_grad_checkpoint(discriminator)
    model.train()
    # discriminator.train()

    # =======================================================
    # 5. boost model for distributed training with colossalai
    # =======================================================
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    torch.set_default_dtype(torch.float)
    num_steps_per_epoch = len(dataloader)
    logger.info("Boost model for distributed training")
    num_steps_per_epoch = len(dataloader)

    # discriminator, disc_optimizer, _, _, disc_lr_scheduler = booster.boost(
    #     model=discriminator, optimizer=disc_optimizer, lr_scheduler=disc_lr_scheduler
    # )
    # logger.info("Boost discriminator for distributed training")

    # =======================================================
    # 6. training loop
    # =======================================================
    start_epoch = start_step = log_step = sampler_start_idx = 0
    acc_step = 0
    running_loss = 0.0
    running_disc_loss = 0.0
    # 6.1. resume training
    if cfg.load is not None:
        logger.info("Loading checkpoint")
        booster.load_model(model, os.path.join(cfg.load, "model"))
        booster.load_optimizer(optimizer, os.path.join(cfg.load, "optimizer"))

        # booster.load_model(discriminator, os.path.join(cfg.load, "discriminator"))
        # booster.load_optimizer(disc_optimizer, os.path.join(cfg.load, "disc_optimizer"))

        # LeCam EMA for discriminator
        # lecam_path = os.path.join(cfg.load, "lecam_states.json")
        # if cfg.lecam_loss_weight is not None and os.path.exists(lecam_path):
        #     lecam_state = load_json(lecam_path)
        #     lecam_ema_real, lecam_ema_fake = lecam_state["lecam_ema_real"], lecam_state["lecam_ema_fake"]
        #     lecam_ema = LeCamEMA(
        #         decay=cfg.ema_decay, ema_real=lecam_ema_real, ema_fake=lecam_ema_fake, dtype=dtype, device=device
        #     )

        running_states = load_json(os.path.join(cfg.load, "running_states.json"))
        dist.barrier()
        start_epoch, start_step, sampler_start_idx = (
            running_states["epoch"],
            running_states["step"],
            running_states["sample_start_index"],
        )
        logger.info(f"Loaded checkpoint {cfg.load} at epoch {start_epoch} step {start_step}")
    logger.info(f"Training for {cfg.epochs} epochs with {num_steps_per_epoch} steps per epoch")

    dataloader.sampler.set_start_index(sampler_start_idx)

    # 6.3. training loop

    # calculate discriminator_time_padding
    # disc_time_downsample_factor = 2 ** len(cfg.discriminator.channel_multipliers)
    # if cfg.dataset.num_frames % disc_time_downsample_factor != 0:
    #     disc_time_padding = disc_time_downsample_factor - cfg.dataset.num_frames % disc_time_downsample_factor
    # else:
    #     disc_time_padding = 0
    # video_contains_first_frame = cfg.video_contains_first_frame

    for epoch in range(start_epoch, cfg.epochs):
        dataloader.sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info(f"Beginning epoch {epoch}...")

        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            total=num_steps_per_epoch,
            initial=start_step,
        ) as pbar:
            for step, batch in pbar:
                x = batch["video"].to(device, dtype)  # [B, C, T, H, W]

                #  ===== Spatial VAE =====
                if cfg.get("vae_2d", None) is not None:
                    with torch.no_grad():
                        x_z = vae_2d.encode(x)

                #  ====== VAE ======
                x_z_rec, posterior, z = model(x_z)
                x_rec = vae_2d.decode(x_z_rec)

                #  ====== Generator Loss ======
                # simple nll loss
                _, weighted_nll_loss, weighted_kl_loss = vae_loss_fn(x, x_rec, posterior)
                _, weighted_z_nll_loss, _ = vae_loss_fn(x_z, x_z_rec, posterior)
                # _, image_identity_loss, _ = vae_loss_fn(x_z, z, posterior)

                # adversarial_loss = torch.tensor(0.0)
                # adversarial loss
                # if global_step > cfg.discriminator_start:
                #     # padded videos for GAN
                #     fake_video = pad_at_dim(recon_video, (disc_time_padding, 0), value=0.0, dim=2)
                #     fake_logits = discriminator(fake_video.contiguous())
                #     adversarial_loss = adversarial_loss_fn(
                #         fake_logits,
                #         nll_loss,
                #         vae.module.get_last_layer(),
                #         global_step,
                #         is_training=vae.training,
                #     )

                # vae_loss = weighted_nll_loss + weighted_kl_loss + adversarial_loss + weighted_z_nll_loss
                # vae_loss = weighted_nll_loss + weighted_kl_loss + weighted_z_nll_loss + image_identity_loss
                vae_loss = weighted_nll_loss + weighted_kl_loss + weighted_z_nll_loss

                optimizer.zero_grad()
                # Backward & update
                booster.backward(loss=vae_loss, optimizer=optimizer)
                # # NOTE: clip gradients? this is done in Open-Sora-Plan
                # torch.nn.utils.clip_grad_norm_(vae.parameters(), 1) # NOTE: done by grad_clip
                optimizer.step()

                # Log loss values:
                all_reduce_mean(vae_loss)  # NOTE: this is to get average loss for logging
                running_loss += vae_loss.item()
                global_step = epoch * num_steps_per_epoch + step
                log_step += 1
                acc_step += 1

                #  ====== Discriminator Loss ======
                # if global_step > cfg.discriminator_start:
                #     # if video_contains_first_frame:
                #     # Since we don't have enough T frames, pad anyways
                #     real_video = pad_at_dim(video, (disc_time_padding, 0), value=0.0, dim=2)
                #     fake_video = pad_at_dim(recon_video, (disc_time_padding, 0), value=0.0, dim=2)

                #     if cfg.gradient_penalty_loss_weight is not None and cfg.gradient_penalty_loss_weight > 0.0:
                #         real_video = real_video.requires_grad_()
                #         real_logits = discriminator(
                #             real_video.contiguous()
                #         )  # SCH: not detached for now for gradient_penalty calculation
                #     else:
                #         real_logits = discriminator(real_video.contiguous().detach())

                #     fake_logits = discriminator(fake_video.contiguous().detach())

                #     lecam_ema_real, lecam_ema_fake = lecam_ema.get()

                #     weighted_d_adversarial_loss, lecam_loss, gradient_penalty_loss = disc_loss_fn(
                #         real_logits,
                #         fake_logits,
                #         global_step,
                #         lecam_ema_real=lecam_ema_real,
                #         lecam_ema_fake=lecam_ema_fake,
                #         real_video=real_video if cfg.gradient_penalty_loss_weight is not None else None,
                #     )
                #     disc_loss = weighted_d_adversarial_loss + lecam_loss + gradient_penalty_loss
                #     if cfg.lecam_loss_weight is not None:
                #         ema_real = torch.mean(real_logits.clone().detach()).to(device, dtype)
                #         ema_fake = torch.mean(fake_logits.clone().detach()).to(device, dtype)
                #         all_reduce_mean(ema_real)
                #         all_reduce_mean(ema_fake)
                #         lecam_ema.update(ema_real, ema_fake)

                #     disc_optimizer.zero_grad()
                #     # Backward & update
                #     booster.backward(loss=disc_loss, optimizer=disc_optimizer)
                #     # # NOTE: TODO: clip gradients? this is done in Open-Sora-Plan
                #     # torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1) # NOTE: done by grad_clip
                #     disc_optimizer.step()

                #     # Log loss values:
                #     all_reduce_mean(disc_loss)
                #     running_disc_loss += disc_loss.item()
                # else:
                #     disc_loss = torch.tensor(0.0)
                #     weighted_d_adversarial_loss = torch.tensor(0.0)
                #     lecam_loss = torch.tensor(0.0)
                #     gradient_penalty_loss = torch.tensor(0.0)

                # Log to tensorboard
                if coordinator.is_master() and (global_step + 1) % cfg.log_every == 0:
                    avg_loss = running_loss / log_step
                    avg_disc_loss = running_disc_loss / log_step
                    pbar.set_postfix(
                        {"loss": avg_loss, "disc_loss": avg_disc_loss, "step": step, "global_step": global_step}
                    )
                    running_loss = 0
                    log_step = 0
                    running_disc_loss = 0
                    writer.add_scalar("loss", vae_loss.item(), global_step)
                    if cfg.wandb:
                        wandb.log(
                            {
                                "iter": global_step,
                                "num_samples": global_step * total_batch_size,
                                "epoch": epoch,
                                "loss": vae_loss.item(),
                                "kl_loss": weighted_kl_loss.item(),
                                # "gen_adv_loss": adversarial_loss.item(),
                                # "disc_loss": disc_loss.item(),
                                # "lecam_loss": lecam_loss.item(),
                                # "r1_grad_penalty": gradient_penalty_loss.item(),
                                "nll_loss": weighted_nll_loss.item(),
                                "z_nll_loss": weighted_z_nll_loss.item(),
                                # "image_identity_loss": image_identity_loss.item(),
                                "avg_loss": avg_loss,
                            },
                            step=global_step,
                        )

                # Save checkpoint
                if cfg.ckpt_every > 0 and (global_step + 1) % cfg.ckpt_every == 0:
                    save_dir = os.path.join(exp_dir, f"epoch{epoch}-global_step{global_step+1}")
                    os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)  # already handled in booster save_model
                    booster.save_model(model, os.path.join(save_dir, "model"), shard=True)
                    # booster.save_model(discriminator, os.path.join(save_dir, "discriminator"), shard=True)
                    booster.save_optimizer(
                        optimizer, os.path.join(save_dir, "optimizer"), shard=True, size_per_shard=4096
                    )
                    # booster.save_optimizer(
                    #     disc_optimizer, os.path.join(save_dir, "disc_optimizer"), shard=True, size_per_shard=4096
                    # )

                    running_states = {
                        "epoch": epoch,
                        "step": step + 1,
                        "global_step": global_step + 1,
                        "sample_start_index": (step + 1) * cfg.batch_size,
                    }

                    # lecam_ema_real, lecam_ema_fake = lecam_ema.get()
                    # lecam_state = {
                    #     "lecam_ema_real": lecam_ema_real.item(),
                    #     "lecam_ema_fake": lecam_ema_fake.item(),
                    # }
                    if coordinator.is_master():
                        save_json(running_states, os.path.join(save_dir, "running_states.json"))
                        # if cfg.lecam_loss_weight is not None:
                        #     save_json(lecam_state, os.path.join(save_dir, "lecam_states.json"))
                    dist.barrier()

                    logger.info(
                        f"Saved checkpoint at epoch {epoch} step {step + 1} global_step {global_step + 1} to {exp_dir}"
                    )

            # print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        dataloader.sampler.set_start_index(0)
        start_step = 0


if __name__ == "__main__":
    main()
