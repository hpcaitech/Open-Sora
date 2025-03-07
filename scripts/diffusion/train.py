import gc
import math
import os
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
import torch.nn.functional as F
import wandb
from colossalai.booster import Booster
from colossalai.utils import set_seed
from peft import LoraConfig
from tqdm import tqdm

from opensora.acceleration.checkpoint import (
    GLOBAL_ACTIVATION_MANAGER,
    set_grad_checkpoint,
)
from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets.aspect import bucket_to_shapes
from opensora.datasets.dataloader import prepare_dataloader
from opensora.datasets.pin_memory_cache import PinMemoryCache
from opensora.models.mmdit.distributed import MMDiTPolicy
from opensora.registry import DATASETS, MODELS, build_module
from opensora.utils.ckpt import (
    CheckpointIO,
    model_sharding,
    record_model_param_shape,
    rm_checkpoints,
)
from opensora.utils.config import (
    config_to_name,
    create_experiment_workspace,
    parse_configs,
)
from opensora.utils.logger import create_logger
from opensora.utils.misc import (
    NsysProfiler,
    Timers,
    all_reduce_mean,
    create_tensorboard_writer,
    is_log_process,
    is_pipeline_enabled,
    log_cuda_max_memory,
    log_cuda_memory,
    log_model_params,
    print_mem,
    to_torch_dtype,
)
from opensora.utils.optimizer import create_lr_scheduler, create_optimizer
from opensora.utils.sampling import (
    get_res_lin_function,
    pack,
    prepare,
    prepare_ids,
    time_shift,
)
from opensora.utils.train import (
    create_colossalai_plugin,
    dropout_condition,
    get_batch_loss,
    prepare_visual_condition_causal,
    prepare_visual_condition_uncausal,
    set_eps,
    set_lr,
    setup_device,
    update_ema,
    warmup_ae,
)

torch.backends.cudnn.benchmark = False  # True leads to slow down in conv3d


def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs()

    # == get dtype & device ==
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    device, coordinator = setup_device()
    grad_ckpt_buffer_size = cfg.get("grad_ckpt_buffer_size", 0)
    if grad_ckpt_buffer_size > 0:
        GLOBAL_ACTIVATION_MANAGER.setup_buffer(grad_ckpt_buffer_size, dtype)
    checkpoint_io = CheckpointIO()
    set_seed(cfg.get("seed", 1024))
    PinMemoryCache.force_dtype = dtype
    pin_memory_cache_pre_alloc_numels = cfg.get("pin_memory_cache_pre_alloc_numels", None)
    PinMemoryCache.pre_alloc_numels = pin_memory_cache_pre_alloc_numels

    # == init ColossalAI booster ==
    plugin_type = cfg.get("plugin", "zero2")
    plugin_config = cfg.get("plugin_config", {})
    plugin_kwargs = {}
    if plugin_type == "hybrid":
        plugin_kwargs["custom_policy"] = MMDiTPolicy
    plugin = create_colossalai_plugin(
        plugin=plugin_type,
        dtype=cfg.get("dtype", "bf16"),
        grad_clip=cfg.get("grad_clip", 0),
        **plugin_config,
        **plugin_kwargs,
    )
    booster = Booster(plugin=plugin)

    seq_align = plugin_config.get("sp_size", 1)

    # == init exp_dir ==
    exp_name, exp_dir = create_experiment_workspace(
        cfg.get("outputs", "./outputs"),
        model_name=config_to_name(cfg),
        config=cfg.to_dict(),
        exp_name=cfg.get("exp_name", None),  # useful for automatic restart to specify the exp_name
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
                name=exp_name,
                config=cfg.to_dict(),
                dir=exp_dir,
            )
    num_gpus = dist.get_world_size() if dist.is_initialized() else 1
    tp_size = cfg["plugin_config"].get("tp_size", 1)
    sp_size = cfg["plugin_config"].get("sp_size", 1)
    pp_size = cfg["plugin_config"].get("pp_size", 1)
    num_groups = num_gpus // (tp_size * sp_size * pp_size)
    logger.info("Number of GPUs: %s", num_gpus)
    logger.info("Number of groups: %s", num_groups)

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
        num_groups=num_groups,
    )
    print_mem("before prepare_dataloader")
    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    print_mem("after prepare_dataloader")
    num_steps_per_epoch = len(dataloader)
    dataset.to_efficient()

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")

    # == build model model ==
    model = build_module(cfg.model, MODELS, device_map=device, torch_dtype=dtype).train()
    if cfg.get("grad_checkpoint", True):
        set_grad_checkpoint(model)
    log_cuda_memory("diffusion")
    log_model_params(model)

    # == build EMA model ==
    use_lora = cfg.get("lora_config", None) is not None
    if cfg.get("ema_decay", None) is not None and not use_lora:
        ema = deepcopy(model).cpu().eval().requires_grad_(False)
        ema_shape_dict = record_model_param_shape(ema)
        logger.info("EMA model created.")
    else:
        ema = ema_shape_dict = None
        logger.info("No EMA model created.")
    log_cuda_memory("EMA")

    # == enable LoRA ==
    if use_lora:
        lora_config = LoraConfig(**cfg.get("lora_config", None))
        model = booster.enable_lora(
            model=model,
            lora_config=lora_config,
            pretrained_dir=cfg.get("lora_checkpoint", None),
        )
        log_cuda_memory("lora")
        log_model_params(model)

    if not cfg.get("cached_video", False):
        # == buildn autoencoder ==
        model_ae = build_module(cfg.ae, MODELS, device_map=device, torch_dtype=dtype).eval().requires_grad_(False)
        del model_ae.decoder
        log_cuda_memory("autoencoder")
        log_model_params(model_ae)
        model_ae.encode = torch.compile(model_ae.encoder, dynamic=True)

    if not cfg.get("cached_text", False):
        # == build text encoder (t5) ==
        model_t5 = build_module(cfg.t5, MODELS, device_map=device, torch_dtype=dtype).eval().requires_grad_(False)
        log_cuda_memory("t5")
        log_model_params(model_t5)

        # == build text encoder (clip) ==
        model_clip = build_module(cfg.clip, MODELS, device_map=device, torch_dtype=dtype).eval().requires_grad_(False)
        log_cuda_memory("clip")
        log_model_params(model_clip)

    # == setup optimizer ==
    optimizer = create_optimizer(model, cfg.optim)

    # == setup lr scheduler ==
    lr_scheduler = create_lr_scheduler(
        optimizer=optimizer,
        num_steps_per_epoch=num_steps_per_epoch,
        epochs=cfg.get("epochs", 1000),
        warmup_steps=cfg.get("warmup_steps", None),
        use_cosine_scheduler=cfg.get("use_cosine_scheduler", False),
    )
    log_cuda_memory("optimizer")

    # == prepare null vectors for dropout ==
    if cfg.get("cached_text", False):
        null_txt = torch.load("/mnt/ddn/sora/tmp_load/null_t5.pt", map_location=device)
        null_vec = torch.load("/mnt/ddn/sora/tmp_load/null_clip.pt", map_location=device)
    else:
        null_txt = model_t5("")
        null_vec = model_clip("")

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
    torch.set_default_dtype(torch.float)
    logger.info("Boosted model for distributed training")
    log_cuda_memory("boost")

    # == global variables ==
    cfg_epochs = cfg.get("epochs", 1000)
    log_step = acc_step = 0
    running_loss = 0.0
    timers = Timers(record_time=cfg.get("record_time", False), record_barrier=cfg.get("record_barrier", False))
    nsys = NsysProfiler(
        warmup_steps=cfg.get("nsys_warmup_steps", 2),
        num_steps=cfg.get("nsys_num_steps", 2),
        enabled=cfg.get("nsys", False),
    )
    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    # == resume ==
    load_master_weights = cfg.get("load_master_weights", False)
    save_master_weights = cfg.get("save_master_weights", False)
    start_epoch = cfg.get("start_epoch", None)
    start_step = cfg.get("start_step", None)
    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint from %s", cfg.load)

        lr_scheduler_to_load = lr_scheduler
        if cfg.get("update_warmup_steps", False):
            lr_scheduler_to_load = None
        ret = checkpoint_io.load(
            booster,
            cfg.load,
            model=model,
            ema=ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler_to_load,
            sampler=(
                None if start_step is not None else sampler
            ),  # if specify start step, set last_micro_batch_access_index of a new sampler instead
            include_master_weights=load_master_weights,
        )
        start_epoch = start_epoch if start_epoch is not None else ret[0]
        start_step = start_step if start_step is not None else ret[1]
        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, ret[0], ret[1])

        # load optimizer and scheduler will overwrite some of the hyperparameters, so we need to reset them
        set_lr(optimizer, lr_scheduler, cfg.optim.lr, cfg.get("initial_lr", None))
        set_eps(optimizer, cfg.optim.eps)

        if cfg.get("update_warmup_steps", False):
            assert (
                cfg.get("warmup_steps", None) is not None
            ), "you need to set warmup_steps in order to pass --update-warmup-steps True"
            # set_warmup_steps(lr_scheduler, cfg.warmup_steps)
            lr_scheduler.step(start_epoch * num_steps_per_epoch + start_step)
            logger.info("The learning rate starts from %s", optimizer.param_groups[0]["lr"])
    if start_step is not None:
        # if start step exceeds data length, go to next epoch
        if start_step > num_steps_per_epoch:
            start_epoch = (
                start_epoch + start_step // num_steps_per_epoch
                if start_epoch is not None
                else start_step // num_steps_per_epoch
            )
            start_step = start_step % num_steps_per_epoch
    else:
        start_step = 0
    sampler.set_step(start_step)
    start_epoch = start_epoch if start_epoch is not None else 0
    logger.info("Starting from epoch %s step %s", start_epoch, start_step)

    # == sharding EMA model ==
    if ema is not None:
        model_sharding(ema)
        ema = ema.to(device)
        log_cuda_memory("sharding EMA")

    # == warmup autoencoder ==
    if cfg.get("warmup_ae", False):
        shapes = bucket_to_shapes(cfg.get("bucket_config", None), batch_size=cfg.ae.batch_size)
        warmup_ae(model_ae, shapes, device, dtype)

    # =======================================================
    # 5. training iter
    # =======================================================
    sigma_min = cfg.get("sigma_min", 1e-5)
    accumulation_steps = cfg.get("accumulation_steps", 1)
    ckpt_every = cfg.get("ckpt_every", 0)

    if cfg.get("is_causal_vae", False):
        prepare_visual_condition = prepare_visual_condition_causal
    else:
        prepare_visual_condition = prepare_visual_condition_uncausal

    @torch.no_grad()
    def prepare_inputs(batch):
        inp = dict()
        x = batch.pop("video")
        y = batch.pop("text")
        bs = x.shape[0]

        # == encode video ==
        with nsys.range("encode_video"), timers["encode_video"]:
            # == prepare condition ==
            if cfg.get("condition_config", None) is not None:
                # condition for i2v & v2v
                x_0, cond = prepare_visual_condition(x, cfg.condition_config, model_ae)
                cond = pack(cond, patch_size=cfg.get("patch_size", 2))
                inp["cond"] = cond
            else:
                if cfg.get("cached_video", False):
                    x_0 = batch.pop("video_latents").to(device=device, dtype=dtype)
                else:
                    x_0 = model_ae.encode(x)

        # == prepare timestep ==
        # follow SD3 time shift, shift_alpha = 1 for 256px and shift_alpha = 3 for 1024px
        shift_alpha = get_res_lin_function()((x_0.shape[-1] * x_0.shape[-2]) // 4)
        # add temporal influence
        shift_alpha *= math.sqrt(x_0.shape[-3])  # for image, T=1 so no effect
        t = torch.sigmoid(torch.randn((bs), device=device))
        t = time_shift(shift_alpha, t).to(dtype)

        if cfg.get("cached_text", False):
            # == encode text ==
            t5_embedding = batch.pop("text_t5").to(device=device, dtype=dtype)
            clip_embedding = batch.pop("text_clip").to(device=device, dtype=dtype)
            with nsys.range("encode_text"), timers["encode_text"]:
                inp_ = prepare_ids(x_0, t5_embedding, clip_embedding)
                inp.update(inp_)
                x_0 = pack(x_0, patch_size=cfg.get("patch_size", 2))
        else:
            # == encode text ==
            with nsys.range("encode_text"), timers["encode_text"]:
                inp_ = prepare(
                    model_t5,
                    model_clip,
                    x_0,
                    prompt=y,
                    seq_align=seq_align,
                    patch_size=cfg.get("patch_size", 2),
                )
                inp.update(inp_)
                x_0 = pack(x_0, patch_size=cfg.get("patch_size", 2))

        # == dropout ==
        if cfg.get("dropout_ratio", None) is not None:
            cur_null_txt = null_txt
            num_pad_null_txt = inp["txt"].shape[1] - cur_null_txt.shape[1]
            if num_pad_null_txt > 0:
                cur_null_txt = torch.cat([cur_null_txt] + [cur_null_txt[:, -1:]] * num_pad_null_txt, dim=1)
            inp["txt"] = dropout_condition(
                cfg.dropout_ratio.get("t5", 0.0),
                inp["txt"],
                cur_null_txt,
            )
            inp["y_vec"] = dropout_condition(
                cfg.dropout_ratio.get("clip", 0.0),
                inp["y_vec"],
                null_vec,
            )

        # == prepare noise vector ==
        x_1 = torch.randn_like(x_0, dtype=torch.float32).to(device, dtype)
        t_rev = 1 - t
        x_t = t_rev[:, None, None] * x_0 + (1 - (1 - sigma_min) * t_rev[:, None, None]) * x_1
        inp["img"] = x_t
        inp["timesteps"] = t.to(dtype)
        inp["guidance"] = torch.full((x_t.shape[0],), cfg.get("guidance", 4), device=x_t.device, dtype=x_t.dtype)

        return inp, x_0, x_1

    def run_iter(inp, x_0, x_1):
        if is_pipeline_enabled(plugin_type, plugin_config):
            inp["target"] = (1 - sigma_min) * x_1 - x_0  # follow MovieGen, modify V_t accordingly
            with nsys.range("forward-backward"), timers["forward-backward"]:
                data_iter = iter([inp])
                if cfg.get("no_i2v_ref_loss", False):
                    loss_fn = (
                        lambda out, input_: get_batch_loss(out, input_["target"], input_.pop("masks", None))
                        / accumulation_steps
                    )
                else:
                    loss_fn = (
                        lambda out, input_: F.mse_loss(out.float(), input_["target"].float(), reduction="mean")
                        / accumulation_steps
                    )
                loss = booster.execute_pipeline(data_iter, model, loss_fn, optimizer)["loss"]
                loss = loss * accumulation_steps if loss is not None else loss
                loss_item = all_reduce_mean(loss.data.clone().detach())
        else:
            with nsys.range("forward"), timers["forward"]:
                model_pred = model(**inp)  # B, T, L
                v_t = (1 - sigma_min) * x_1 - x_0
                if cfg.get("no_i2v_ref_loss", False):
                    loss = get_batch_loss(model_pred, v_t, inp.pop("masks", None))
                else:
                    loss = F.mse_loss(model_pred.float(), v_t.float(), reduction="mean")

            loss_item = all_reduce_mean(loss.data.clone().detach()).item()

            # == backward & update ==
            dist.barrier()
            with nsys.range("backward"), timers["backward"]:
                ctx = (
                    booster.no_sync(model, optimizer)
                    if cfg.get("plugin", "zero2") in ("zero1", "zero1-seq") and (step + 1) % accumulation_steps != 0
                    else nullcontext()
                )
                with ctx:
                    booster.backward(loss=(loss / accumulation_steps), optimizer=optimizer)

        with nsys.range("optim"), timers["optim"]:
            if (step + 1) % accumulation_steps == 0:
                booster.checkpoint_io.synchronize()
                optimizer.step()
                optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()

        # == update EMA ==
        if ema is not None:
            with nsys.range("update_ema"), timers["update_ema"]:
                update_ema(
                    ema,
                    model.unwrap(),
                    optimizer=optimizer,
                    decay=cfg.get("ema_decay", 0.9999),
                )

        return loss_item

    # =======================================================
    # 6. training loop
    # =======================================================
    dist.barrier()
    for epoch in range(start_epoch, cfg_epochs):
        # == set dataloader to new epoch ==
        sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info("Beginning epoch %s...", epoch)

        # == training loop in an epoch ==
        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not is_log_process(plugin_type, plugin_config),
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            pbar_iter = iter(pbar)

            # prefetch one for non-blocking data loading
            def fetch_data():
                step, batch = next(pbar_iter)
                # print(f"==debug== rank{dist.get_rank()} {dataloader_iter.get_cache_info()}")
                pinned_video = batch["video"]
                batch["video"] = pinned_video.to(device, dtype, non_blocking=True)
                return batch, step, pinned_video

            batch_, step_, pinned_video_ = fetch_data()

            for _ in range(start_step, num_steps_per_epoch):
                nsys.step()
                # == load data ===
                with nsys.range("load_data"), timers["load_data"]:
                    batch, step, pinned_video = batch_, step_, pinned_video_

                    if step + 1 < num_steps_per_epoch:
                        # only fetch new data if not last step
                        batch_, step_, pinned_video_ = fetch_data()

                # == run iter ==
                with nsys.range("iter"), timers["iter"]:
                    inp, x_0, x_1 = prepare_inputs(batch)
                    if cache_pin_memory:
                        dataloader_iter.remove_cache(pinned_video)
                    loss = run_iter(inp, x_0, x_1)

                # == update log info ==
                if loss is not None:
                    running_loss += loss

                # == log config ==
                global_step = epoch * num_steps_per_epoch + step
                actual_update_step = (global_step + 1) // accumulation_steps
                log_step += 1
                acc_step += 1

                # == logging ==
                if (global_step + 1) % accumulation_steps == 0:
                    if actual_update_step % cfg.get("log_every", 1) == 0:
                        if is_log_process(plugin_type, plugin_config):
                            avg_loss = running_loss / log_step
                            # progress bar
                            pbar.set_postfix(
                                {
                                    "loss": avg_loss,
                                    "global_grad_norm": optimizer.get_grad_norm(),
                                    "step": step,
                                    "global_step": global_step,
                                    # "actual_update_step": actual_update_step,
                                    "lr": optimizer.param_groups[0]["lr"],
                                }
                            )
                            # tensorboard
                            if tb_writer is not None:
                                tb_writer.add_scalar("loss", loss, actual_update_step)
                            # wandb
                            if cfg.get("wandb", False):
                                wandb_dict = {
                                    "iter": global_step,
                                    "acc_step": acc_step,
                                    "epoch": epoch,
                                    "loss": loss,
                                    "avg_loss": avg_loss,
                                    "lr": optimizer.param_groups[0]["lr"],
                                    "eps": optimizer.param_groups[0]["eps"],
                                    "global_grad_norm": optimizer.get_grad_norm(),  # test grad norm
                                }
                                if cfg.get("record_time", False):
                                    wandb_dict.update(timers.to_dict())
                                wandb.log(wandb_dict, step=actual_update_step)

                        running_loss = 0.0
                        log_step = 0

                # == checkpoint saving ==
                # uncomment below 3 lines to forcely clean cache
                with nsys.range("clean_cache"), timers["clean_cache"]:
                    if ckpt_every > 0 and actual_update_step % ckpt_every == 0 and coordinator.is_master():
                        subprocess.run("sudo drop_cache", shell=True)

                with nsys.range("checkpoint"), timers["checkpoint"]:
                    if ckpt_every > 0 and actual_update_step % ckpt_every == 0:
                        # mannual garbage collection
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
                            lora=use_lora,
                            actual_update_step=actual_update_step,
                            ema_shape_dict=ema_shape_dict,
                            async_io=cfg.get("async_io", False),
                            include_master_weights=save_master_weights,
                        )

                        if is_log_process(plugin_type, plugin_config):
                            os.system(f"chgrp -R share {save_dir}")

                        logger.info(
                            "Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                            epoch,
                            step + 1,
                            actual_update_step,
                            save_dir,
                        )

                        # remove old checkpoints
                        rm_checkpoints(exp_dir, keep_n_latest=cfg.get("keep_n_latest", -1))
                        logger.info("Removed old checkpoints and kept %s latest ones.", cfg.get("keep_n_latest", -1))
                # uncomment below 3 lines to benchmark checkpoint
                # if ckpt_every > 0 and actual_update_step % ckpt_every == 0:
                #     booster.checkpoint_io._sync_io()
                #     checkpoint_io._sync_io()
                # == terminal timer ==
                if cfg.get("record_time", False):
                    print(timers.to_str(epoch, step))

        sampler.reset()
        start_step = 0
    log_cuda_max_memory("final")


if __name__ == "__main__":
    main()
