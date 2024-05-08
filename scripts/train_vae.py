import os
import random
from datetime import timedelta
from pprint import pformat

import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from einops import rearrange
from tqdm import tqdm

import wandb
from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets import prepare_dataloader
from opensora.models.vae.losses import AdversarialLoss, DiscriminatorLoss, VAELoss
from opensora.registry import DATASETS, MODELS, build_module
from opensora.utils.ckpt_utils import create_logger, load_json, save_json
from opensora.utils.config_utils import (
    create_tensorboard_writer,
    define_experiment_workspace,
    parse_configs,
    save_training_config,
)
from opensora.utils.misc import all_reduce_mean, format_numel_str, get_model_numel, to_torch_dtype
from opensora.utils.train_utils import create_colossalai_plugin

DEFAULT_DATASET_NAME = "VideoTextDataset"


def main():
    # ======================================================
    # 1. configs & runtime variables & colossalai launch
    # ======================================================
    cfg = parse_configs(training=True)

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

    # == colossalai init distributed training ==
    # NOTE: A very large timeout is set to avoid some processes exit early
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(1024)
    coordinator = DistCoordinator()
    device = get_current_device()

    # == init exp_dir ==
    exp_name, exp_dir = define_experiment_workspace(cfg)
    coordinator.block_all()
    if coordinator.is_master():
        os.makedirs(exp_dir, exist_ok=True)
        save_training_config(cfg.to_dict(), exp_dir)
    coordinator.block_all()

    # == init logger, tensorboard & wandb ==
    if coordinator.is_master():
        logger = create_logger(exp_dir)
        logger.info(f"Experiment directory created at {exp_dir}")
        logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
        tb_writer = create_tensorboard_writer(exp_dir)
        if cfg.get("wandb", False):
            wandb.init(project="minisora", name=exp_name, config=cfg.to_dict())
    else:
        logger = create_logger(None)

    # == init ColossalAI booster ==
    plugin = create_colossalai_plugin(
        plugin=cfg.get("plugin", "zero2"),
        dtype=cfg_dtype,
        grad_clip=cfg.get("grad_clip", 0),
        sp_size=cfg.get("sp_size", 1),
    )
    booster = Booster(plugin=plugin)

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info("Dataset contains %s samples.", len(dataset))
    # == build dataloader ==
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
    if cfg.dataset.type == DEFAULT_DATASET_NAME:
        dataloader = prepare_dataloader(**dataloader_args)
        total_batch_size = cfg.batch_size * dist.get_world_size() // cfg.get("sp_size", 1)
        logger.info("Total batch size: %s", total_batch_size)
    else:
        dataloader = prepare_variable_dataloader(
            bucket_config=cfg.bucket_config,
            num_bucket_build_workers=cfg.num_bucket_build_workers,
            **dataloader_args,
        )

    # ======================================================
    # 3. build model
    # ======================================================
    # == build vae model ==
    model = build_module(cfg.model, MODELS).to(device, dtype)
    model.train()
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        "Trainable model params: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel),
    )
    # == build discriminator model ==
    if cfg.get("discriminator", False) != False:
        discriminator = build_module(cfg.discriminator, MODELS).to(device, dtype)
        discriminator.train()
        discriminator_numel, discriminator_numel_trainable = get_model_numel(discriminator)
        logger.info(
            "Trainable model params: %s, Total model params: %s",
            format_numel_str(discriminator_numel_trainable),
            format_numel_str(discriminator_numel),
        )

    # == setup loss functions ==
    vae_loss_fn = VAELoss(
        logvar_init=cfg.get("logvar_init", 0.0),
        perceptual_loss_weight=cfg.get("perceptual_loss_weight", 0.1),
        kl_loss_weight=cfg.get("kl_loss_weight", 1e-6),
        device=device,
        dtype=dtype,
    )

    if cfg.get("discriminator", False) != False:
        adversarial_loss_fn = AdversarialLoss(
            discriminator_factor=cfg.get("discriminator_factor", 1),
            discriminator_start=cfg.get("discriminator_start", -1),
            generator_factor=cfg.get("generator_factor", 0.5),
            generator_loss_type=cfg.get("generator_loss_type", "hinge"),
        )

        disc_loss_fn = DiscriminatorLoss(
            discriminator_factor=cfg.get("discriminator_factor", 1),
            discriminator_start=cfg.get("discriminator_start", -1),
            discriminator_loss_type=cfg.get("discriminator_loss_type", "hinge"),
            lecam_loss_weight=cfg.get("lecam_loss_weight", None),
            gradient_penalty_loss_weight=cfg.get("gradient_penalty_loss_weight", None),
        )

    # == setup vae optimizer ==
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        adamw_mode=True,
        lr=cfg.get("lr", 1e-5),
        weight_decay=cfg.get("weight_decay", 0),
    )
    lr_scheduler = None

    # == setup discriminator optimizer ==
    if cfg.get("discriminator", False) != False:
        disc_optimizer = HybridAdam(
            filter(lambda p: p.requires_grad, discriminator.parameters()),
            adamw_mode=True,
            lr=cfg.get("lr", 1e-5),
            weight_decay=cfg.get("weight_decay", 0),
        )
        disc_lr_scheduler = None

    # == additional preparation ==
    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(model)

    # =======================================================
    # 4. distributed training preparation with colossalai
    # =======================================================
    # == boosting ==
    # NOTE: we set dtype first to make initialization of model consistent with the dtype; then reset it to the fp32 as we make diffusion scheduler in fp32
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
    logger.info("Boosting model for distributed training")
    if cfg.dataset.type == DEFAULT_DATASET_NAME:
        num_steps_per_epoch = len(dataloader)
    else:
        num_steps_per_epoch = dataloader.batch_sampler.get_num_batch() // dist.get_world_size()
        None if cfg.get("start_from_scratch ", False) else dataloader.batch_sampler

    cfg_epochs = cfg.get("epochs", 1000)
    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    # == global variables ==
    start_epoch = start_step = log_step = sampler_start_idx = acc_step = 0
    running_loss = 0.0
    running_disc_loss = 0.0

    # == resume ==
    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint")
        booster.load_model(model, os.path.join(cfg.load, "model"))
        booster.load_optimizer(optimizer, os.path.join(cfg.load, "optimizer"))
        if cfg.get("discriminator", False) != False and os.path.exists(os.path.join(cfg.load, "discriminator")):
            booster.load_model(discriminator, os.path.join(cfg.load, "discriminator"))
            booster.load_optimizer(disc_optimizer, os.path.join(cfg.load, "disc_optimizer"))
        running_states = load_json(os.path.join(cfg.load, "running_states.json"))
        start_epoch, start_step, sampler_start_idx = (
            running_states["epoch"],
            running_states["step"],
            running_states["sample_start_index"],
        )
        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, start_epoch, start_step)
        dist.barrier()

    if cfg.dataset.type == DEFAULT_DATASET_NAME:
        dataloader.sampler.set_start_index(sampler_start_idx)

    # =======================================================
    # 5. training loop
    # =======================================================
    for epoch in range(start_epoch, cfg_epochs):
        # == set dataloader to new epoch ==
        if cfg.dataset.type == DEFAULT_DATASET_NAME:
            dataloader.sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info("Beginning epoch %s...", epoch)

        # == training loop in an epoch ==
        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            total=num_steps_per_epoch,
            initial=start_step,
        ) as pbar:
            for step, batch in pbar:
                x = batch["video"].to(device, dtype)  # [B, C, T, H, W]
                # if random.random() < cfg.get("mixed_image_ratio", 0.0):
                #     x = x[:, :, :1, :, :]
                length = random.randint(1, x.size(2))
                x = x[:, :, :length, :, :]

                # == vae encoding ===
                x_rec, x_z_rec, z, posterior, x_z = model(x)

                # == loss initialization ==
                vae_loss = torch.tensor(0.0, device=device, dtype=dtype)
                disc_loss = torch.tensor(0.0, device=device, dtype=dtype)
                log_dict = {}

                # == real image reconstruction loss computation ==
                nll_loss, weighted_nll_loss, weighted_kl_loss = vae_loss_fn(x, x_rec, posterior)
                log_dict["kl_loss"] = weighted_kl_loss.item()
                log_dict["nll_loss"] = weighted_nll_loss.item()
                if cfg.get("use_real_rec_loss", False):
                    vae_loss += weighted_nll_loss + weighted_kl_loss

                # == temporal vae reconstruction loss computation ==
                _, weighted_z_nll_loss, _ = vae_loss_fn(x_z, x_z_rec, posterior, no_perceptual=True)
                log_dict["z_nll_loss"] = weighted_z_nll_loss.item()

                # == use z reconstruction loss ==
                if cfg.get("use_z_rec_loss", False):
                    vae_loss += weighted_z_nll_loss

                # == use image loss only ==
                if cfg.get("use_image_identity_loss", False) and x.size(2) == 1:
                    _, image_identity_loss, _ = vae_loss_fn(x_z, z, posterior, no_perceptual=True)
                    vae_loss += image_identity_loss
                    log_dict["image_identity_loss"] = image_identity_loss.item()

                # == generator adversarial loss ==
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

                # == generator backward & update ==
                optimizer.zero_grad()
                booster.backward(loss=vae_loss, optimizer=optimizer)
                optimizer.step()

                # == discriminator adversarial loss ==
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

                    # == discriminator backward & update ==
                    disc_optimizer.zero_grad()
                    booster.backward(loss=disc_loss, optimizer=disc_optimizer)
                    disc_optimizer.step()
                    all_reduce_mean(disc_loss)
                    running_disc_loss += disc_loss.item()

                # == update log info ==
                all_reduce_mean(vae_loss)
                running_loss += vae_loss.item()
                global_step = epoch * num_steps_per_epoch + step
                log_step += 1
                acc_step += 1

                # == logging ==
                if coordinator.is_master() and (global_step + 1) % cfg.log_every == 0:
                    avg_loss = running_loss / log_step
                    avg_disc_loss = running_disc_loss / log_step
                    pbar.set_postfix(
                        {"loss": avg_loss, "disc_loss": avg_disc_loss, "step": step, "global_step": global_step}
                    )
                    # tensorboard
                    tb_writer.add_scalar("loss", vae_loss.item(), global_step)
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
                    running_loss = 0
                    running_disc_loss = 0
                    log_step = 0

                # == checkpoint saving ==
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
                        "Saved checkpoint at epoch %s step %s global_step %s to %s",
                        epoch,
                        step + 1,
                        global_step + 1,
                        exp_dir,
                    )

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        if cfg.dataset.type == DEFAULT_DATASET_NAME:
            dataloader.sampler.set_start_index(0)
        else:
            dataloader.batch_sampler.set_epoch(epoch + 1)
            logger.info("Epoch done, recomputing batch sampler")
        start_step = 0


if __name__ == "__main__":
    main()
