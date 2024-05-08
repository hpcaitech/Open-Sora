import os
from copy import deepcopy
from datetime import timedelta
from pprint import pformat

import torch
import torch.distributed as dist
import wandb
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from tqdm import tqdm

from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets import prepare_dataloader, prepare_variable_dataloader
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import create_logger, load, model_sharding, record_model_param_shape, save
from opensora.utils.config_utils import (
    create_tensorboard_writer,
    define_experiment_workspace,
    parse_configs,
    save_training_config,
)
from opensora.utils.misc import all_reduce_mean, format_numel_str, get_model_numel, requires_grad, to_torch_dtype
from opensora.utils.train_utils import MaskGenerator, create_colossalai_plugin, update_ema

DEFAULT_DATASET_NAME = "VideoTextDataset"


def main():
    # ======================================================
    # 1. configs & runtime variables & colossalai launch
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=True)

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
    if coordinator.is_master():
        logger = create_logger(exp_dir)
        logger.info("Experiment directory created at %s", exp_dir)
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
    # == build dataset ==
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
            bucket_config=cfg.get("bucket_config", None),
            num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
            **dataloader_args,
        )

    # ======================================================
    # 3. build model
    # ======================================================
    # == build text-encoder and vae ==
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype)
    vae.eval()

    # == build diffusion model ==
    input_size = (dataset.num_frames, *dataset.image_size)
    latent_size = vae.get_latent_size(input_size)
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
    ).to(device, dtype)
    model.train()
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        "Trainable model params: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel),
    )

    # == build ema for diffusion model ==
    ema = deepcopy(model).to(torch.float32).to(device)
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)
    ema.eval()
    update_ema(ema, model, decay=0, sharded=False)

    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # == setup optimizer ==
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        adamw_mode=True,
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("adam_eps", 1e-8),
    )
    lr_scheduler = None

    # == additional preparation ==
    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(model)
    if cfg.get("mask_ratios", None) is not None:
        mask_generator = MaskGenerator(cfg.mask_ratios)

    # =======================================================
    # 5. distributed training preparation with colossalai
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
    torch.set_default_dtype(torch.float)
    logger.info("Boosting model for distributed training")
    if cfg.dataset.type == DEFAULT_DATASET_NAME:
        num_steps_per_epoch = len(dataloader)
        sampler_to_io = None
    else:
        num_steps_per_epoch = dataloader.batch_sampler.get_num_batch() // dist.get_world_size()
        sampler_to_io = None if cfg.get("start_from_scratch ", False) else dataloader.batch_sampler

    cfg_epochs = cfg.get("epochs", 1000)
    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    # == global variables ==
    start_epoch = start_step = log_step = sampler_start_idx = acc_step = 0
    running_loss = 0.0

    # == resume ==
    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint")
        ret = load(
            booster,
            model,
            ema,
            optimizer,
            lr_scheduler,
            cfg.load,
            sampler=sampler_to_io,
        )
        if not cfg.get("start_from_scratch ", False):
            start_epoch, start_step, sampler_start_idx = ret
        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, start_epoch, start_step)
    if cfg.dataset.type == DEFAULT_DATASET_NAME:
        dataloader.sampler.set_start_index(sampler_start_idx)

    model_sharding(ema)

    # =======================================================
    # 6. training loop
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
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            for step, batch in pbar:
                x = batch.pop("video").to(device, dtype)  # [B, C, T, H, W]
                y = batch.pop("text")

                # == visual and text encoding ==
                with torch.no_grad():
                    # Prepare visual inputs
                    x = vae.encode(x)  # [B, C, T, H/P, W/P]
                    # Prepare text inputs
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
                loss_dict = scheduler.training_losses(model, x, model_args, mask=mask)

                # == backward & update ==
                loss = loss_dict["loss"].mean()
                booster.backward(loss=loss, optimizer=optimizer)
                optimizer.step()
                optimizer.zero_grad()

                # == update EMA ==
                update_ema(ema, model.module, optimizer=optimizer, decay=cfg.get("ema_decay", 0.9999))

                # == update log info ==
                all_reduce_mean(loss)
                running_loss += loss.item()
                global_step = epoch * num_steps_per_epoch + step
                log_step += 1
                acc_step += 1

                # == logging ==
                if coordinator.is_master() and (global_step + 1) % cfg.log_every == 0:
                    avg_loss = running_loss / log_step
                    # progress bar
                    pbar.set_postfix({"loss": avg_loss, "step": step, "global_step": global_step})
                    # tensorboard
                    tb_writer.add_scalar("loss", loss.item(), global_step)
                    # wandb
                    if cfg.wandb:
                        wandb.log(
                            {
                                "iter": global_step,
                                "epoch": epoch,
                                "loss": loss.item(),
                                "avg_loss": avg_loss,
                                "acc_step": acc_step,
                            },
                            step=global_step,
                        )

                    running_loss = 0
                    log_step = 0

                # == checkpoint saving ==
                if cfg.ckpt_every > 0 and (global_step + 1) % cfg.ckpt_every == 0:
                    save(
                        booster,
                        model,
                        ema,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        step + 1,
                        global_step + 1,
                        cfg.batch_size,
                        coordinator,
                        exp_dir,
                        ema_shape_dict,
                        sampler=sampler_to_io,
                    )
                    logger.info(
                        "Saved checkpoint at epoch %s step %s global_step %s to %s",
                        epoch,
                        step + 1,
                        global_step + 1,
                        exp_dir,
                    )

        # NOTE: the continue epochs are not resumed, so we need to reset the sampler start index and start step
        if cfg.dataset.type == DEFAULT_DATASET_NAME:
            dataloader.sampler.set_start_index(0)
        else:
            dataloader.batch_sampler.set_epoch(epoch + 1)
            logger.info("Epoch done, recomputing batch sampler")
        start_step = 0


if __name__ == "__main__":
    main()
