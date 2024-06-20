import os
import random
from datetime import timedelta
from pprint import pformat

import torch
import torch.distributed as dist
import wandb
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from einops import rearrange
from tqdm import tqdm

from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets.dataloader import prepare_dataloader
from opensora.models.vae.losses import AdversarialLoss, DiscriminatorLoss, VAELoss
from opensora.registry import DATASETS, MODELS, build_module
from opensora.utils.ckpt_utils import load, save
from opensora.utils.config_utils import define_experiment_workspace, parse_configs, save_training_config
from opensora.utils.misc import (
    all_reduce_mean,
    create_logger,
    create_tensorboard_writer,
    format_numel_str,
    get_model_numel,
    to_torch_dtype,
)
from opensora.utils.train_utils import create_colossalai_plugin


def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=True)

    # == device and dtype ==
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

    # == colossalai init distributed training ==
    # NOTE: A very large timeout is set to avoid some processes exit early
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(cfg.get("seed", 1024))
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
    logger = create_logger(exp_dir)
    logger.info("Experiment directory created at %s", exp_dir)
    logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    if coordinator.is_master():
        tb_writer = create_tensorboard_writer(exp_dir)
        if cfg.get("wandb", False):
            wandb.init(project="minisora", name=exp_name, config=cfg.to_dict(), dir="./outputs/wandb")

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
    logger.info("Building dataset...")
    # == build dataset ==
    assert cfg.dataset.type == "VideoTextDataset", "Only support VideoTextDataset for vae training"
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )
    dataloader, sampler = prepare_dataloader(**dataloader_args)
    total_batch_size = cfg.batch_size * dist.get_world_size() // cfg.get("sp_size", 1)
    logger.info("Total batch size: %s", total_batch_size)
    num_steps_per_epoch = len(dataloader)

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")
    # == build vae model ==
    model = build_module(cfg.model, MODELS).to(device, dtype).train()
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        "[VAE] Trainable model params: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel),
    )

    # == build discriminator model ==
    use_discriminator = cfg.get("discriminator", None) is not None
    if use_discriminator:
        discriminator = build_module(cfg.discriminator, MODELS).to(device, dtype).train()
        discriminator_numel, discriminator_numel_trainable = get_model_numel(discriminator)
        logger.info(
            "[Discriminator] Trainable model params: %s, Total model params: %s",
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

    if use_discriminator:
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
    if use_discriminator:
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
    logger.info("Preparing for distributed training...")
    # == boosting ==
    # NOTE: we set dtype first to make initialization of model consistent with the dtype; then reset it to the fp32 as we make diffusion scheduler in fp32
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    if use_discriminator:
        discriminator, disc_optimizer, _, _, disc_lr_scheduler = booster.boost(
            model=discriminator,
            optimizer=disc_optimizer,
            lr_scheduler=disc_lr_scheduler,
        )
    torch.set_default_dtype(torch.float)
    logger.info("Boosting model for distributed training")

    # == global variables ==
    cfg_epochs = cfg.get("epochs", 1000)
    start_epoch = start_step = log_step = sampler_start_idx = acc_step = 0
    running_loss = running_disc_loss = 0.0
    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    # == resume ==
    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint")
        start_epoch, start_step = load(
            booster,
            cfg.load,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            sampler=sampler,
        )
        if use_discriminator and os.path.exists(os.path.join(cfg.load, "discriminator")):
            booster.load_model(discriminator, os.path.join(cfg.load, "discriminator"))
            booster.load_optimizer(disc_optimizer, os.path.join(cfg.load, "disc_optimizer"))
        dist.barrier()
        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, start_epoch, start_step)

    # =======================================================
    # 5. training loop
    # =======================================================
    dist.barrier()
    for epoch in range(start_epoch, cfg_epochs):
        # == set dataloader to new epoch ==
        sampler.set_epoch(epoch)
        dataiter = iter(dataloader)
        logger.info("Beginning epoch %s...", epoch)

        # == training loop in an epoch ==
        with tqdm(
            enumerate(dataiter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            total=num_steps_per_epoch,
            initial=start_step,
        ) as pbar:
            for step, batch in pbar:
                x = batch["video"].to(device, dtype)  # [B, C, T, H, W]

                # == mixed training setting ==
                mixed_strategy = cfg.get("mixed_strategy", None)
                if mixed_strategy == "mixed_video_image":
                    if random.random() < cfg.get("mixed_image_ratio", 0.0):
                        x = x[:, :, :1, :, :]
                elif mixed_strategy == "mixed_video_random":
                    length = random.randint(1, x.size(2))
                    x = x[:, :, :length, :, :]

                # == vae encoding & decoding ===
                x_rec, x_z_rec, z, posterior, x_z = model(x)

                # == loss initialization ==
                vae_loss = torch.tensor(0.0, device=device, dtype=dtype)
                disc_loss = torch.tensor(0.0, device=device, dtype=dtype)
                log_dict = {}

                # == loss: real image reconstruction ==
                nll_loss, weighted_nll_loss, weighted_kl_loss = vae_loss_fn(x, x_rec, posterior)
                log_dict["kl_loss"] = weighted_kl_loss.item()
                log_dict["nll_loss"] = weighted_nll_loss.item()
                if cfg.get("use_real_rec_loss", False):
                    vae_loss += weighted_nll_loss + weighted_kl_loss

                # == loss: temporal vae reconstruction ==
                _, weighted_z_nll_loss, _ = vae_loss_fn(x_z, x_z_rec, posterior, no_perceptual=True)
                log_dict["z_nll_loss"] = weighted_z_nll_loss.item()
                if cfg.get("use_z_rec_loss", False):
                    vae_loss += weighted_z_nll_loss

                # == loss: image only distillation ==
                if cfg.get("use_image_identity_loss", False) and x.size(2) == 1:
                    _, image_identity_loss, _ = vae_loss_fn(x_z, z, posterior, no_perceptual=True)
                    vae_loss += image_identity_loss
                    log_dict["image_identity_loss"] = image_identity_loss.item()

                # == loss: generator adversarial ==
                if use_discriminator:
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
                    log_dict["adversarial_loss"] = adversarial_loss.item()
                    vae_loss += adversarial_loss

                # == generator backward & update ==
                optimizer.zero_grad()
                booster.backward(loss=vae_loss, optimizer=optimizer)
                optimizer.step()
                all_reduce_mean(vae_loss)
                running_loss += vae_loss.item()

                # == loss: discriminator adversarial ==
                if use_discriminator:
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
                    log_dict["disc_loss"] = disc_loss.item()

                    # == discriminator backward & update ==
                    disc_optimizer.zero_grad()
                    booster.backward(loss=disc_loss, optimizer=disc_optimizer)
                    disc_optimizer.step()
                    all_reduce_mean(disc_loss)
                    running_disc_loss += disc_loss.item()

                # == update log info ==
                global_step = epoch * num_steps_per_epoch + step
                log_step += 1
                acc_step += 1

                # == logging ==
                if coordinator.is_master() and (global_step + 1) % cfg.get("log_every", 1) == 0:
                    avg_loss = running_loss / log_step
                    avg_disc_loss = running_disc_loss / log_step
                    # progress bar
                    pbar.set_postfix(
                        {"loss": avg_loss, "disc_loss": avg_disc_loss, "step": step, "global_step": global_step}
                    )
                    # tensorboard
                    tb_writer.add_scalar("loss", vae_loss.item(), global_step)
                    # wandb
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
                    running_loss = running_disc_loss = 0.0
                    log_step = 0

                # == checkpoint saving ==
                ckpt_every = cfg.get("ckpt_every", 0)
                if ckpt_every > 0 and (global_step + 1) % ckpt_every == 0:
                    save(
                        booster,
                        exp_dir,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        epoch=epoch,
                        step=step + 1,
                        global_step=global_step + 1,
                        batch_size=cfg.get("batch_size", None),
                        sampler=sampler,
                    )

                    save_dir = os.path.join(exp_dir, f"epoch{epoch}-global_step{global_step+1}")
                    if use_discriminator:
                        booster.save_model(discriminator, os.path.join(save_dir, "discriminator"), shard=True)
                        booster.save_optimizer(
                            disc_optimizer, os.path.join(save_dir, "disc_optimizer"), shard=True, size_per_shard=4096
                        )
                    dist.barrier()

                    logger.info(
                        "Saved checkpoint at epoch %s step %s global_step %s to %s",
                        epoch,
                        step + 1,
                        global_step + 1,
                        exp_dir,
                    )

        sampler.reset()
        start_step = 0


if __name__ == "__main__":
    main()
