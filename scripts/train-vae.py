import os
import random
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
from einops import rearrange

import wandb
from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import get_data_parallel_group, set_data_parallel_group
from opensora.datasets import prepare_dataloader
from opensora.models.vae.losses import VAELoss, AdversarialLoss, DiscriminatorLoss
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
    total_batch_size = cfg.batch_size * dist.get_world_size()
    logger.info(f"Total batch size: {total_batch_size}")

    # ======================================================
    # 4. build model
    # ======================================================
    # 4.1. build model
    model = build_module(cfg.model, MODELS)
    model.to(device, dtype)
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        f"Trainable model params: {format_numel_str(model_numel_trainable)}, Total model params: {format_numel_str(model_numel)}"
    )

    # 4.4 loss functions
    vae_loss_fn = VAELoss(
        logvar_init=cfg.get("logvar_init", 0.0),
        perceptual_loss_weight=cfg.perceptual_loss_weight,
        kl_loss_weight=cfg.kl_loss_weight,
        device=device,
        dtype=dtype,
    )

    if cfg.get("discriminator", False) != False:
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

    # 4.6. prepare for training
    if cfg.grad_checkpoint:
        set_grad_checkpoint(model)
    model.train()


    # 4.7 add discriminator if specified in config
    if cfg.get("discriminator", False) != False:
        discriminator = build_module(cfg.discriminator, MODELS)
        discriminator.to(device, dtype)
        discriminator_numel, discriminator_numel_trainable = get_model_numel(discriminator)
        logger.info(
            f"Trainable model params: {format_numel_str(discriminator_numel_trainable)}, Total model params: {format_numel_str(discriminator_numel)}"
        )
        disc_optimizer = HybridAdam(
            filter(lambda p: p.requires_grad, discriminator.parameters()), lr=cfg.lr, weight_decay=0, adamw_mode=True
        )
        disc_lr_scheduler = None
        discriminator.train()

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
    if cfg.get("discriminator", False) != False:
        discriminator, disc_optimizer, _, _, disc_lr_scheduler = booster.boost(
            model=discriminator,
            optimizer=disc_optimizer,
            lr_scheduler=disc_lr_scheduler,
        )

    torch.set_default_dtype(torch.float)
    num_steps_per_epoch = len(dataloader)
    logger.info("Boost model for distributed training")
    num_steps_per_epoch = len(dataloader)

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

        running_states = load_json(os.path.join(cfg.load, "running_states.json"))

        if cfg.get("discriminator", False) != False and os.path.exists(os.path.join(cfg.load, "discriminator")):
            booster.load_model(discriminator, os.path.join(cfg.load, "discriminator"))
            booster.load_optimizer(disc_optimizer, os.path.join(cfg.load, "disc_optimizer"))
        
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
                if random.random() < cfg.get("mixed_image_ratio", 0.0):
                    x = x[:, :, :1, :, :]

                #  ===== VAE =====
                x_rec, x_z_rec, z, posterior, x_z = model(x)

                #  ====== Generator Loss ======
                vae_loss = torch.tensor(0.0, device=device, dtype=dtype)
                disc_loss = torch.tensor(0.0, device=device, dtype=dtype)

                log_dict = {}

                # real image reconstruction loss
                nll_loss, weighted_nll_loss, weighted_kl_loss = vae_loss_fn(x, x_rec, posterior)
                log_dict["kl_loss"] = weighted_kl_loss.item()
                log_dict["nll_loss"] = weighted_nll_loss.item()
                if cfg.get("use_real_rec_loss", False):
                    vae_loss += weighted_nll_loss + weighted_kl_loss

                _, weighted_z_nll_loss, _ = vae_loss_fn(x_z, x_z_rec, posterior, no_perceptual=True)
                log_dict["z_nll_loss"] = weighted_z_nll_loss.item()
                # z reconstruction loss
                if cfg.get("use_z_rec_loss", False):
                    vae_loss += weighted_z_nll_loss

                # only for image
                if cfg.get("use_image_identity_loss", False) and x.size(2) == 1:
                    _, image_identity_loss, _ = vae_loss_fn(x_z, z, posterior, no_perceptual=True)
                    vae_loss += image_identity_loss
                    log_dict["image_identity_loss"] = image_identity_loss.item()

                # Adversarial Generator Loss
                if cfg.get("discriminator", False) != False:
                    recon_video = rearrange(x_rec, "b c t h w -> (b t) c h w").contiguous()
                    global_step = epoch * num_steps_per_epoch + step
                    fake_logits = discriminator(recon_video.contiguous())
                    adversarial_loss = adversarial_loss_fn(
                        fake_logits,
                        nll_loss,
                        model.module.get_temporal_last_layer(),
                        global_step,
                        is_training=model.training,
                    )
                    vae_loss += adversarial_loss

                # Backward & update
                booster.backward(loss=vae_loss, optimizer=optimizer)
                optimizer.step()
                optimizer.zero_grad()


                # Adversarial Discriminator loss
                if cfg.get("discriminator", False) != False:
                    real_video = rearrange(x, "b c t h w -> (b t) c h w").contiguous()
                    fake_video = rearrange(x_rec, "b c t h w -> (b t) c h w").contiguous()
                    real_logits = discriminator(real_video.contiguous().detach())
                    fake_logits = discriminator(fake_video.contiguous().detach())
                    weighted_d_adversarial_loss, _, _ = disc_loss_fn(
                            real_logits,
                            fake_logits,
                            global_step, 
                    )
                    disc_loss = weighted_d_adversarial_loss
                    # Backward & update
                    booster.backward(loss=disc_loss, optimizer=disc_optimizer)
                    disc_optimizer.step()
                    disc_optimizer.zero_grad()
                    all_reduce_mean(disc_loss)
                    running_disc_loss += disc_loss.item()

                # Log loss values:
                all_reduce_mean(vae_loss)
                running_loss += vae_loss.item()
                global_step = epoch * num_steps_per_epoch + step
                log_step += 1
                acc_step += 1

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
                                "avg_loss": avg_loss,
                                **log_dict,
                            },
                            step=global_step,
                        )

                # Save checkpoint
                if cfg.ckpt_every > 0 and (global_step + 1) % cfg.ckpt_every == 0:
                    save_dir = os.path.join(exp_dir, f"epoch{epoch}-global_step{global_step+1}")
                    os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)  # already handled in booster save_model
                    booster.save_model(model, os.path.join(save_dir, "model"), shard=True)
                    booster.save_optimizer(
                        optimizer, os.path.join(save_dir, "optimizer"), shard=True, size_per_shard=4096
                    )
                    if cfg.get("discriminator", False) != False:
                        booster.save_model(discriminator, os.path.join(save_dir, "discriminator"), shard=True)
                        booster.save_optimizer(
                            disc_optimizer, os.path.join(save_dir, "disc_optimizer"), shard=True, size_per_shard=4096
                        )

                    running_states = {
                        "epoch": epoch,
                        "step": step + 1,
                        "global_step": global_step + 1,
                        "sample_start_index": (step + 1) * cfg.batch_size,
                    }
                    if coordinator.is_master():
                        save_json(running_states, os.path.join(save_dir, "running_states.json"))
                    dist.barrier()
                    logger.info(
                        f"Saved checkpoint at epoch {epoch} step {step + 1} global_step {global_step + 1} to {exp_dir}"
                    )

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        dataloader.sampler.set_start_index(0)
        start_step = 0


if __name__ == "__main__":
    main()
