import gc
import os
import random
import subprocess
import warnings
from contextlib import nullcontext
from copy import deepcopy
from pprint import pformat

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
gc.disable()


import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.utils import set_seed
from torch.profiler import ProfilerActivity, profile, schedule
from tqdm import tqdm

import wandb
from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets.dataloader import prepare_dataloader
from opensora.datasets.pin_memory_cache import PinMemoryCache
from opensora.models.vae.losses import DiscriminatorLoss, GeneratorLoss, VAELoss
from opensora.registry import DATASETS, MODELS, build_module
from opensora.utils.ckpt import CheckpointIO, model_sharding, record_model_param_shape, rm_checkpoints
from opensora.utils.config import config_to_name, create_experiment_workspace, parse_configs
from opensora.utils.logger import create_logger
from opensora.utils.misc import (
    Timer,
    all_reduce_sum,
    create_tensorboard_writer,
    is_log_process,
    log_model_params,
    to_torch_dtype,
)
from opensora.utils.optimizer import create_lr_scheduler, create_optimizer
from opensora.utils.train import create_colossalai_plugin, set_lr, set_warmup_steps, setup_device, update_ema

torch.backends.cudnn.benchmark = True

WAIT = 1
WARMUP = 10
ACTIVE = 20

my_schedule = schedule(
    wait=WAIT,  # number of warmup steps
    warmup=WARMUP,  # number of warmup steps with profiling
    active=ACTIVE,  # number of active steps with profiling
)


def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs()

    # == get dtype & device ==
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    device, coordinator = setup_device()
    checkpoint_io = CheckpointIO()
    set_seed(cfg.get("seed", 1024))
    PinMemoryCache.force_dtype = dtype
    pin_memory_cache_pre_alloc_numels = cfg.get("pin_memory_cache_pre_alloc_numels", None)
    PinMemoryCache.pre_alloc_numels = pin_memory_cache_pre_alloc_numels

    # == init ColossalAI booster ==
    plugin_type = cfg.get("plugin", "zero2")
    plugin_config = cfg.get("plugin_config", {})
    plugin = (
        create_colossalai_plugin(
            plugin=plugin_type,
            dtype=cfg.get("dtype", "bf16"),
            grad_clip=cfg.get("grad_clip", 0),
            **plugin_config,
        )
        if plugin_type != "none"
        else None
    )
    booster = Booster(plugin=plugin)

    # == init exp_dir ==
    exp_name, exp_dir = create_experiment_workspace(
        cfg.get("outputs", "./outputs"),
        model_name=config_to_name(cfg),
        config=cfg.to_dict(),
    )
    if is_log_process(plugin_type, plugin_config):
        print(f"changing {exp_dir} to share")
        os.system(f"chgrp -R share {exp_dir}")

    # == init logger, tensorboard & wandb ==
    logger = create_logger(exp_dir)
    logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    tb_writer = None
    if coordinator.is_master():
        tb_writer = create_tensorboard_writer(exp_dir)
        if cfg.get("wandb", False):
            wandb.init(
                project=cfg.get("wandb_project", "Open-Sora"),
                name=cfg.get("wandb_expr_name", exp_name),
                config=cfg.to_dict(),
                dir=exp_dir,
            )

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")
    # == build dataset ==
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    cache_pin_memory = pin_memory_cache_pre_alloc_numels is not None
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", None),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
        cache_pin_memory=cache_pin_memory,
    )
    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    num_steps_per_epoch = len(dataloader)

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")

    # == build vae model ==
    model = build_module(cfg.model, MODELS, device_map=device, torch_dtype=dtype).train()
    log_model_params(model)

    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(model)
    vae_loss_fn = VAELoss(**cfg.vae_loss_config, device=device, dtype=dtype)

    # == build EMA model ==
    if cfg.get("ema_decay", None) is not None:
        ema = deepcopy(model).cpu().eval().requires_grad_(False)
        ema_shape_dict = record_model_param_shape(ema)
        logger.info("EMA model created.")
    else:
        ema = ema_shape_dict = None
        logger.info("No EMA model created.")

    # == build discriminator model ==
    use_discriminator = cfg.get("discriminator", None) is not None
    if use_discriminator:
        discriminator = build_module(cfg.discriminator, MODELS).to(device, dtype).train()
        log_model_params(discriminator)
        generator_loss_fn = GeneratorLoss(**cfg.gen_loss_config)
        discriminator_loss_fn = DiscriminatorLoss(**cfg.disc_loss_config)

    # == setup optimizer ==
    optimizer = create_optimizer(model, cfg.optim)

    # == setup lr scheduler ==
    lr_scheduler = create_lr_scheduler(
        optimizer=optimizer, num_steps_per_epoch=num_steps_per_epoch, epochs=cfg.get("epochs", 1000), **cfg.lr_scheduler
    )

    # == setup discriminator optimizer ==
    if use_discriminator:
        disc_optimizer = create_optimizer(discriminator, cfg.optim_discriminator)
        disc_lr_scheduler = create_lr_scheduler(
            optimizer=disc_optimizer,
            num_steps_per_epoch=num_steps_per_epoch,
            epochs=cfg.get("epochs", 1000),
            **cfg.disc_lr_scheduler,
        )

    # =======================================================
    # 4. distributed training preparation with colossalai
    # =======================================================
    logger.info("Preparing for distributed training...")
    # == boosting ==
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
    logger.info("Boosted model for distributed training")

    # == global variables ==
    cfg_epochs = cfg.get("epochs", 1000)
    mixed_strategy = cfg.get("mixed_strategy", None)
    mixed_image_ratio = cfg.get("mixed_image_ratio", 0.0)
    # modulate mixed image ratio since we force rank 0 to be video
    num_ranks = dist.get_world_size()
    modulated_mixed_image_ratio = (
        num_ranks * mixed_image_ratio / (num_ranks - 1) if num_ranks > 1 else mixed_image_ratio
    )
    if is_log_process(plugin_type, plugin_config):
        print("modulated mixed image ratio:", modulated_mixed_image_ratio)

    start_epoch = start_step = log_step = acc_step = 0
    running_loss = dict(  # loss accumulated over config.log_every steps
        all=0.0,
        nll=0.0,
        nll_rec=0.0,
        nll_per=0.0,
        kl=0.0,
        gen=0.0,
        gen_w=0.0,
        disc=0.0,
        debug=0.0,
    )

    def log_loss(name, loss, loss_dict, use_video):
        # only calculate loss for video
        if use_video == 0:
            loss.data = torch.tensor(0.0, device=device, dtype=dtype)
        all_reduce_sum(loss.data)
        num_video = torch.tensor(use_video, device=device, dtype=dtype)
        all_reduce_sum(num_video)
        loss_item = loss.item() / num_video.item()
        loss_dict[name] = loss_item
        running_loss[name] += loss_item

    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    # == resume ==
    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint from %s", cfg.load)
        start_epoch = cfg.get("start_epoch", None)
        start_step = cfg.get("start_step", None)
        ret = checkpoint_io.load(
            booster,
            cfg.load,
            model=model,
            ema=ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            sampler=(
                None if start_step is not None else sampler
            ),  # if specify start step, set last_micro_batch_access_index of a new sampler instead
        )
        if start_step is not None:
            # if start step exceeds data length, go to next epoch
            if start_step > num_steps_per_epoch:
                start_epoch = (
                    start_epoch + start_step // num_steps_per_epoch
                    if start_epoch is not None
                    else start_step // num_steps_per_epoch
                )
                start_step = start_step % num_steps_per_epoch
            sampler.set_step(start_step)

        start_epoch = start_epoch if start_epoch is not None else ret[0]
        start_step = start_step if start_step is not None else ret[1]

        if (
            use_discriminator
            and os.path.exists(os.path.join(cfg.load, "discriminator"))
            and not cfg.get("restart_disc", False)
        ):
            booster.load_model(discriminator, os.path.join(cfg.load, "discriminator"))
            if cfg.get("load_optimizer", True):
                booster.load_optimizer(disc_optimizer, os.path.join(cfg.load, "disc_optimizer"))
                if disc_lr_scheduler is not None:
                    booster.load_lr_scheduler(disc_lr_scheduler, os.path.join(cfg.load, "disc_lr_scheduler"))
                if cfg.get("disc_lr", None) is not None:
                    set_lr(disc_optimizer, disc_lr_scheduler, cfg.disc_lr)

        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, start_epoch, start_step)

        if cfg.get("lr", None) is not None:
            set_lr(optimizer, lr_scheduler, cfg.lr, cfg.get("initial_lr", None))

        if cfg.get("update_warmup_steps", False):
            assert (
                cfg.lr_scheduler.get("warmup_steps", None) is not None
            ), "you need to set lr_scheduler.warmup_steps in order to pass --update-warmup-steps True"
            set_warmup_steps(lr_scheduler, cfg.lr_scheduler.warmup_steps)
            if use_discriminator:
                assert (
                    cfg.disc_lr_scheduler.get("warmup_steps", None) is not None
                ), "you need to set disc_lr_scheduler.warmup_steps in order to pass --update-warmup-steps True"
                set_warmup_steps(disc_lr_scheduler, cfg.disc_lr_scheduler.warmup_steps)

    # == sharding EMA model ==
    if ema is not None:
        model_sharding(ema)
        ema = ema.to(device)

    if cfg.get("freeze_layers", None) == "all":
        for param in model.module.parameters():
            param.requires_grad = False
        print("all layers frozen")

    # model.module.requires_grad_(False)
    # =======================================================
    # 5. training loop
    # =======================================================
    dist.barrier()
    accumulation_steps = int(cfg.get("accumulation_steps", 1))
    for epoch in range(start_epoch, cfg_epochs):
        # == set dataloader to new epoch ==
        sampler.set_epoch(epoch)
        dataiter = iter(dataloader)
        logger.info("Beginning epoch %s...", epoch)
        random.seed(1024 + dist.get_rank())  # load vid/img for each rank

        # == training loop in an epoch ==
        with tqdm(
            enumerate(dataiter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            total=num_steps_per_epoch,
            initial=start_step,
        ) as pbar:
            pbar_iter = iter(pbar)

            def fetch_data():
                step, batch = next(pbar_iter)
                pinned_video = batch["video"]
                batch["video"] = pinned_video.to(device, dtype, non_blocking=True)
                return batch, step, pinned_video

            batch_, step_, pinned_video_ = fetch_data()

            profiler_ctxt = (
                profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=my_schedule,
                    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/profile"),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                )
                if cfg.get("profile", False)
                else nullcontext()
            )

            with profiler_ctxt:
                for _ in range(start_step, num_steps_per_epoch):
                    if cfg.get("profile", False) and _ == WARMUP + ACTIVE + WAIT + 3:
                        break

                    # == load data ===
                    batch, step, pinned_video = batch_, step_, pinned_video_
                    if step + 1 < num_steps_per_epoch:
                        batch_, step_, pinned_video_ = fetch_data()

                    # == log config ==
                    global_step = epoch * num_steps_per_epoch + step
                    actual_update_step = (global_step + 1) // accumulation_steps
                    log_step += 1
                    acc_step += 1

                    # == mixed strategy ==
                    x = batch["video"]
                    t_length = x.size(2)
                    use_video = 1
                    if mixed_strategy == "mixed_video_image":
                        if random.random() < modulated_mixed_image_ratio and dist.get_rank() != 0:
                            # NOTE: enable the first rank to use video
                            t_length = 1
                            use_video = 0
                    elif mixed_strategy == "mixed_video_random":
                        t_length = random.randint(1, x.size(2))
                    x = x[:, :, :t_length, :, :]

                    with Timer("model", log=True) if cfg.get("profile", False) else nullcontext():
                        # == forward pass ==
                        x_rec, posterior, z = model(x)

                        if cfg.get("profile", False):
                            profiler_ctxt.step()

                        if cache_pin_memory:
                            dataiter.remove_cache(pinned_video)

                        # == loss initialization ==
                        vae_loss = torch.tensor(0.0, device=device, dtype=dtype)
                        loss_dict = {}  # loss at every step

                        # == reconstruction loss ==
                        ret = vae_loss_fn(x, x_rec, posterior)
                        nll_loss = ret["nll_loss"]
                        kl_loss = ret["kl_loss"]
                        recon_loss = ret["recon_loss"]
                        perceptual_loss = ret["perceptual_loss"]
                        vae_loss += nll_loss + kl_loss

                        # == generator loss ==
                        if use_discriminator:
                            # turn off grad update for disc
                            discriminator.requires_grad_(False)
                            fake_logits = discriminator(x_rec.contiguous())

                            generator_loss, g_loss = generator_loss_fn(
                                fake_logits,
                                nll_loss,
                                model.module.get_last_layer(),
                                actual_update_step,
                                is_training=model.training,
                            )
                            # print(f"generator_loss: {generator_loss}, recon_loss: {recon_loss}, perceptual_loss: {perceptual_loss}")

                            vae_loss += generator_loss
                            # turn on disc training
                            discriminator.requires_grad_(True)

                        # == generator backward & update ==
                        ctx = (
                            booster.no_sync(model, optimizer)
                            if cfg.get("plugin", "zero2") in ("zero1", "zero1-seq")
                            and (step + 1) % accumulation_steps != 0
                            else nullcontext()
                        )
                        with Timer("backward", log=True) if cfg.get("profile", False) else nullcontext():
                            with ctx:
                                booster.backward(loss=vae_loss / accumulation_steps, optimizer=optimizer)

                        with Timer("optimizer", log=True) if cfg.get("profile", False) else nullcontext():
                            if (step + 1) % accumulation_steps == 0:
                                optimizer.step()
                                optimizer.zero_grad()
                                if lr_scheduler is not None:
                                    lr_scheduler.step(
                                        actual_update_step,
                                    )
                                # == update EMA ==
                                if ema is not None:
                                    update_ema(
                                        ema,
                                        model.unwrap(),
                                        optimizer=optimizer,
                                        decay=cfg.get("ema_decay", 0.9999),
                                    )

                    # == logging ==
                    log_loss("all", vae_loss, loss_dict, use_video)
                    log_loss("nll", nll_loss, loss_dict, use_video)
                    log_loss("nll_rec", recon_loss, loss_dict, use_video)
                    log_loss("nll_per", perceptual_loss, loss_dict, use_video)
                    log_loss("kl", kl_loss, loss_dict, use_video)
                    if use_discriminator:
                        log_loss("gen_w", generator_loss, loss_dict, use_video)
                        log_loss("gen", g_loss, loss_dict, use_video)

                    # == loss: discriminator adversarial ==
                    if use_discriminator:
                        real_logits = discriminator(x.detach().contiguous())
                        fake_logits = discriminator(x_rec.detach().contiguous())
                        disc_loss = discriminator_loss_fn(
                            real_logits,
                            fake_logits,
                            actual_update_step,
                        )

                        # == discriminator backward & update ==
                        ctx = (
                            booster.no_sync(discriminator, disc_optimizer)
                            if cfg.get("plugin", "zero2") in ("zero1", "zero1-seq")
                            and (step + 1) % accumulation_steps != 0
                            else nullcontext()
                        )
                        with ctx:
                            booster.backward(loss=disc_loss / accumulation_steps, optimizer=disc_optimizer)
                        if (step + 1) % accumulation_steps == 0:
                            disc_optimizer.step()
                            disc_optimizer.zero_grad()
                            if disc_lr_scheduler is not None:
                                disc_lr_scheduler.step(actual_update_step)

                        # log
                        log_loss("disc", disc_loss, loss_dict, use_video)

                    # == logging ==
                    if (global_step + 1) % accumulation_steps == 0:
                        if coordinator.is_master() and actual_update_step % cfg.get("log_every", 1) == 0:
                            avg_loss = {k: v / log_step for k, v in running_loss.items()}
                            # progress bar
                            pbar.set_postfix(
                                {
                                    # "step": step,
                                    # "global_step": global_step,
                                    # "actual_update_step": actual_update_step,
                                    # "lr": optimizer.param_groups[0]["lr"],
                                    **{k: f"{v:.2f}" for k, v in avg_loss.items()},
                                }
                            )
                            # tensorboard
                            tb_writer.add_scalar("loss", vae_loss.item(), actual_update_step)
                            # wandb
                            if cfg.get("wandb", False):
                                wandb.log(
                                    {
                                        "iter": global_step,
                                        "epoch": epoch,
                                        "lr": optimizer.param_groups[0]["lr"],
                                        "avg_loss_": avg_loss,
                                        "avg_loss": avg_loss["all"],
                                        "loss_": loss_dict,
                                        "loss": vae_loss.item(),
                                        "global_grad_norm": optimizer.get_grad_norm(),
                                    },
                                    step=actual_update_step,
                                )

                            running_loss = {k: 0.0 for k in running_loss}
                            log_step = 0

                        # == checkpoint saving ==
                        ckpt_every = cfg.get("ckpt_every", 0)
                        if ckpt_every > 0 and actual_update_step % ckpt_every == 0 and coordinator.is_master():
                            subprocess.run("sudo drop_cache", shell=True)

                        if ckpt_every > 0 and actual_update_step % ckpt_every == 0:
                            # mannually garbage collection
                            gc.collect()

                            save_dir = checkpoint_io.save(
                                booster,
                                exp_dir,
                                model=model,
                                ema=ema,
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                sampler=sampler,
                                epoch=epoch,
                                step=step + 1,
                                global_step=global_step + 1,
                                batch_size=cfg.get("batch_size", None),
                                actual_update_step=actual_update_step,
                                ema_shape_dict=ema_shape_dict,
                                async_io=True,
                            )

                            if is_log_process(plugin_type, plugin_config):
                                os.system(f"chgrp -R share {save_dir}")

                            if use_discriminator:
                                booster.save_model(discriminator, os.path.join(save_dir, "discriminator"), shard=True)
                                booster.save_optimizer(
                                    disc_optimizer,
                                    os.path.join(save_dir, "disc_optimizer"),
                                    shard=True,
                                    size_per_shard=4096,
                                )
                                if disc_lr_scheduler is not None:
                                    booster.save_lr_scheduler(
                                        disc_lr_scheduler, os.path.join(save_dir, "disc_lr_scheduler")
                                    )
                            dist.barrier()

                            logger.info(
                                "Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                                epoch,
                                step + 1,
                                actual_update_step,
                                save_dir,
                            )

                            # remove old checkpoints
                            rm_checkpoints(exp_dir, keep_n_latest=cfg.get("keep_n_latest", -1))
                            logger.info(
                                "Removed old checkpoints and kept %s latest ones.", cfg.get("keep_n_latest", -1)
                            )

            if cfg.get("profile", False):
                profiler_ctxt.export_chrome_trace("./log/profile/trace.json")

        sampler.reset()
        start_step = 0


if __name__ == "__main__":
    main()
