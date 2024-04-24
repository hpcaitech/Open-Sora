from copy import deepcopy

import colossalai
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.distributed as dist
import wandb
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
from tqdm import tqdm
import os
from einops import rearrange
import numpy as np

from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)
from opensora.acceleration.plugin import ZeroSeqParallelPlugin
from opensora.datasets import DatasetFromCSV, get_transforms_image, get_transforms_video, prepare_dataloader
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import create_logger, load_json, save_json, load, model_sharding, record_model_param_shape, save
from opensora.utils.config_utils import (
    create_experiment_workspace,
    create_tensorboard_writer,
    parse_configs,
    save_training_config,
)
from opensora.utils.misc import all_reduce_mean, format_numel_str, get_model_numel, requires_grad, to_torch_dtype
from opensora.utils.train_utils import update_ema, MaskGenerator
from opensora.models.vae.vae_3d_v2 import VEALoss, DiscriminatorLoss, AdversarialLoss, LeCamEMA, pad_at_dim



# efficiency
# from torch.profiler import profile, record_function, ProfilerActivity

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=5)
    print(output)
    # p.export_chrome_trace("/home/shenchenhui/Open-Sora-dev/outputs/traces/trace_" + str(p.step_num) + ".json")


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
    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()
    device = get_current_device()
    dtype = to_torch_dtype(cfg.dtype)

    # 2.2. init logger, tensorboard & wandb
    if not coordinator.is_master():
        logger = create_logger(None)
    else:
        print(cfg)
        logger = create_logger(exp_dir)
        logger.info(f"Experiment directory created at {exp_dir}")

        writer = create_tensorboard_writer(exp_dir)
        if cfg.wandb:
            # wandb.init(project="opensora-vae", name=exp_name, config=cfg._cfg_dict)
            # NOTE: here we use the outputs folder name to store running records of different experiments (since frequent interruption)
            name = os.path.basename(os.path.normpath(cfg.outputs))
            wandb.init(project="opensora-vae", name=name, config=cfg._cfg_dict)

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
    dataset = DatasetFromCSV(
        cfg.data_path,
        transform=(
            get_transforms_video(cfg.image_size[0])
            if not cfg.use_image_transform
            else get_transforms_image(cfg.image_size[0])
        ),
        num_frames=cfg.num_frames,
        frame_interval=cfg.frame_interval,
        root=cfg.root,
    )

    dataloader = prepare_dataloader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )
    logger.info(f"Dataset contains {len(dataset):,} videos ({cfg.data_path})")

    total_batch_size = cfg.batch_size * dist.get_world_size() // cfg.sp_size
    logger.info(f"Total batch size: {total_batch_size}")

    # ======================================================
    # 4. build model
    # ======================================================
    # 4.1. build model

    if cfg.get("use_pipeline") == True:
        # use 2D VAE, then temporal VAE
        vae_2d = build_module(cfg.vae_2d, MODELS)

    vae = build_module(cfg.model, MODELS, device=device)
    vae_numel, vae_numel_trainable = get_model_numel(vae)
    logger.info(
        f"Trainable vae params: {format_numel_str(vae_numel_trainable)}, Total model params: {format_numel_str(vae_numel)}"
    )
    
    discriminator = build_module(cfg.discriminator, MODELS, device=device)
    discriminator_numel, discriminator_numel_trainable = get_model_numel(discriminator)
    logger.info(
        f"Trainable discriminator params: {format_numel_str(discriminator_numel_trainable)}, Total model params: {format_numel_str(discriminator_numel)}"
    )

    # 4.3. move to device
    if cfg.get("use_pipeline") == True:
        vae_2d.to(device, dtype).eval() # eval mode, not training!

    vae = vae.to(device, dtype)
    discriminator = discriminator.to(device, dtype)


    # 4.5. setup optimizer
    # vae optimizer
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, vae.parameters()), lr=cfg.lr, weight_decay=0, adamw_mode=True
    )
    lr_scheduler = None
    # disc optimizer
    disc_optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, discriminator.parameters()), lr=cfg.lr, weight_decay=0, adamw_mode=True
    )
    disc_lr_scheduler = None

    # 4.6. prepare for training
    if cfg.grad_checkpoint:
        set_grad_checkpoint(vae)
        set_grad_checkpoint(discriminator)
    vae.train()
    discriminator.train()


    # =======================================================
    # 5. boost model for distributed training with colossalai
    # =======================================================
    torch.set_default_dtype(dtype)
    vae, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=vae, optimizer=optimizer, lr_scheduler=lr_scheduler, dataloader=dataloader
    )
    torch.set_default_dtype(torch.float)
    num_steps_per_epoch = len(dataloader)
    logger.info("Boost vae for distributed training")


    discriminator, disc_optimizer, _, _, disc_lr_scheduler = booster.boost(
        model=discriminator, optimizer=disc_optimizer, lr_scheduler=disc_lr_scheduler
    )
    logger.info("Boost discriminator for distributed training")


    # =======================================================
    # 6. training loop
    # =======================================================
    start_epoch = start_step = log_step = sampler_start_idx = 0
    running_loss = 0.0
    running_disc_loss = 0.0


    # 6.1. resume training
    if cfg.load is not None:
        logger.info("Loading checkpoint")
        booster.load_model(vae, os.path.join(cfg.load, "model"))
        booster.load_model(discriminator, os.path.join(cfg.load, "discriminator"))
        booster.load_optimizer(optimizer, os.path.join(cfg.load, "optimizer"))
        booster.load_optimizer(disc_optimizer, os.path.join(cfg.load, "disc_optimizer"))
        if lr_scheduler is not None:
            booster.load_lr_scheduler(lr_scheduler, os.path.join(cfg.load, "lr_scheduler"))
        if disc_lr_scheduler is not None:
            booster.load_lr_scheduler(disc_lr_scheduler, os.path.join(cfg.load, "disc_lr_scheduler"))

        running_states = load_json(os.path.join(cfg.load, "running_states.json"))
        dist.barrier()
        start_epoch, start_step, sampler_start_idx = running_states["epoch"], running_states["step"], running_states["sample_start_index"]
        logger.info(f"Loaded checkpoint {cfg.load} at epoch {start_epoch} step {start_step}")
    logger.info(f"Training for {cfg.epochs} epochs with {num_steps_per_epoch} steps per epoch")

    dataloader.sampler.set_start_index(sampler_start_idx)

    # 6.2 Define loss functions
    vae_loss_fn = VEALoss(
        logvar_init=cfg.logvar_init,
        perceptual_loss_weight = cfg.perceptual_loss_weight,
        kl_loss_weight = cfg.kl_loss_weight,
        device=device,
        dtype=dtype,
    )

    adversarial_loss_fn = AdversarialLoss(
        discriminator_factor = cfg.discriminator_factor,
        discriminator_start = cfg.discriminator_start,
        generator_factor = cfg.generator_factor,
        generator_loss_type = cfg.generator_loss_type,
    )

    disc_loss_fn = DiscriminatorLoss(
        discriminator_factor = cfg.discriminator_factor,
        discriminator_start = cfg.discriminator_start,
        discriminator_loss_type = cfg.discriminator_loss_type,
        lecam_loss_weight = cfg.lecam_loss_weight,
        gradient_penalty_loss_weight = cfg.gradient_penalty_loss_weight,
    )

    # 6.3. training loop

    # calculate discriminator_time_padding
    disc_time_downsample_factor = 2 ** len(cfg.discriminator.channel_multipliers)
    if cfg.num_frames % disc_time_downsample_factor != 0:
        disc_time_padding = disc_time_downsample_factor - cfg.num_frames % disc_time_downsample_factor
    else:
        disc_time_padding = 0
    video_contains_first_frame = cfg.video_contains_first_frame

    
    # lecam_ema_real = torch.tensor(0.0)
    # lecam_ema_fake = torch.tensor(0.0)
    lecam_ema = LeCamEMA(decay=cfg.ema_decay, dtype=dtype, device=device)

    for epoch in range(start_epoch, cfg.epochs):
        dataloader.sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info(f"Beginning epoch {epoch}...")



        with tqdm(
            range(start_step, num_steps_per_epoch),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            total=num_steps_per_epoch,
            initial=start_step,
        ) as pbar:
            
            # with profile(
            #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            #     schedule=torch.profiler.schedule(
            #             wait=1,
            #             warmup=1,
            #             active=2,
            #             repeat=2,
            #         ),
            #     on_trace_ready=torch.profiler.tensorboard_trace_handler('/home/shenchenhui/log'),
            #     with_stack=True,
            #     record_shapes=True,
            #     profile_memory=True,
            # ) as p: # trace efficiency
                
                for step in pbar:

                    # with profile(
                    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    #     with_stack=True,
                    # ) as p: # trace efficiency

                        # SCH: calc global step at the start
                        global_step = epoch * num_steps_per_epoch + step
                    
                        batch = next(dataloader_iter)
                        x = batch["video"].to(device, dtype)  # [B, C, T, H, W]

                        # supprt for image or video inputs
                        assert x.ndim in {4, 5}, f"received input of {x.ndim} dimensions" # either image or video
                        assert x.shape[-2:] == cfg.image_size, f"received input size {x.shape[-2:]}, but config image size is {cfg.image_size}"
                        is_image = x.ndim == 4
                        if is_image:
                            video = rearrange(x, 'b c ... -> b c 1 ...')
                            video_contains_first_frame = True
                        else:
                            video = x
                        
                        #  ===== Spatial VAE =====
                        if cfg.get("use_pipeline") == True:
                            with torch.no_grad():
                                video = vae_2d.encode(video)

                        #  ====== VAE ======
                        recon_video, posterior = vae(
                            video,
                            video_contains_first_frame = video_contains_first_frame,
                        )

                        #  ====== Generator Loss ======
                        # simple nll loss
                        nll_loss, weighted_nll_loss, weighted_kl_loss = vae_loss_fn(
                            video,
                            recon_video,
                            posterior,
                            split = "train"
                        )

                        adversarial_loss = torch.tensor(0.0)
                        # adversarial loss 
                        if global_step > cfg.discriminator_start:
                            # padded videos for GAN
                            fake_video = pad_at_dim(recon_video, (disc_time_padding, 0), value = 0., dim = 2)
                            fake_logits = discriminator(fake_video.contiguous())
                            adversarial_loss = adversarial_loss_fn(
                                fake_logits,
                                nll_loss, 
                                vae.module.get_last_layer(),
                                global_step,
                                is_training = vae.training,
                            )
                        
                        vae_loss = weighted_nll_loss + weighted_kl_loss + adversarial_loss

                        optimizer.zero_grad()
                        # Backward & update 
                        booster.backward(loss=vae_loss, optimizer=optimizer)
                        # # NOTE: clip gradients? this is done in Open-Sora-Plan
                        # torch.nn.utils.clip_grad_norm_(vae.parameters(), 1) # NOTE: done by grad_clip
                        optimizer.step()

                        # Log loss values:
                        all_reduce_mean(vae_loss) # NOTE: this is to get average loss for logging
                        running_loss += vae_loss.item()
                        

                        #  ====== Discriminator Loss ======
                        if global_step > cfg.discriminator_start:
                            # if video_contains_first_frame:
                            # Since we don't have enough T frames, pad anyways
                            real_video = pad_at_dim(video, (disc_time_padding, 0), value = 0., dim = 2)
                            fake_video = pad_at_dim(recon_video, (disc_time_padding, 0), value = 0., dim = 2)

                            if cfg.gradient_penalty_loss_weight is not None and cfg.gradient_penalty_loss_weight > 0.0:
                                real_video = real_video.requires_grad_()
                                real_logits = discriminator(real_video.contiguous()) # SCH: not detached for now for gradient_penalty calculation
                            else:
                                real_logits = discriminator(real_video.contiguous().detach()) 

                            fake_logits = discriminator(fake_video.contiguous().detach())


                            lecam_ema_real, lecam_ema_fake = lecam_ema.get()

                            weighted_d_adversarial_loss, lecam_loss, gradient_penalty_loss = disc_loss_fn(
                                real_logits, 
                                fake_logits, 
                                global_step, 
                                lecam_ema_real = lecam_ema_real, 
                                lecam_ema_fake = lecam_ema_fake, 
                                real_video = real_video if cfg.gradient_penalty_loss_weight is not None else None,
                            )
                            disc_loss = weighted_d_adversarial_loss + lecam_loss + gradient_penalty_loss
                            if cfg.ema_decay is not None: 
                                # SCH: TODO: is this written properly like this for moving average? e.g. distributed training etc.
                                # lecam_ema_real = lecam_ema_real * cfg.ema_decay + (1 - cfg.ema_decay) * torch.mean(real_logits.clone().detach())
                                # lecam_ema_fake = lecam_ema_fake * cfg.ema_decay + (1 - cfg.ema_decay) * torch.mean(fake_logits.clone().detach())
                                ema_real = torch.mean(real_logits.clone().detach()).to(device, dtype)
                                ema_fake = torch.mean(fake_logits.clone().detach()).to(device, dtype)
                                all_reduce_mean(ema_real)
                                all_reduce_mean(ema_fake)
                                lecam_ema.update(ema_real, ema_fake)

                            disc_optimizer.zero_grad()
                            # Backward & update
                            booster.backward(loss=disc_loss, optimizer=disc_optimizer)
                            # # NOTE: TODO: clip gradients? this is done in Open-Sora-Plan
                            # torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1) # NOTE: done by grad_clip
                            disc_optimizer.step()

                            # Log loss values:
                            all_reduce_mean(disc_loss)
                            running_disc_loss += disc_loss.item()
                        else:
                            disc_loss = torch.tensor(0.0)
                            weighted_d_adversarial_loss = torch.tensor(0.0)
                            lecam_loss = torch.tensor(0.0)
                            gradient_penalty_loss = torch.tensor(0.0)

                        log_step += 1

                        # Log to tensorboard
                        if coordinator.is_master() and (global_step + 1) % cfg.log_every == 0:
                            avg_loss = running_loss / log_step
                            avg_disc_loss = running_disc_loss / log_step
                            pbar.set_postfix({"loss": avg_loss, "disc_loss": avg_disc_loss, "step": step, "global_step": global_step})
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
                                        "gen_adv_loss": adversarial_loss.item(),
                                        "disc_loss": disc_loss.item(),
                                        "lecam_loss": lecam_loss.item(),
                                        "r1_grad_penalty": gradient_penalty_loss.item(),
                                        "nll_loss": weighted_nll_loss.item(),
                                        "avg_loss": avg_loss,
                                    },
                                    step=global_step,
                                )

                        # Save checkpoint
                        if cfg.ckpt_every > 0 and (global_step + 1) % cfg.ckpt_every == 0:
                            save_dir = os.path.join(exp_dir, f"epoch{epoch}-global_step{global_step+1}")
                            os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)
                            booster.save_model(vae, os.path.join(save_dir, "model"), shard=True)
                            booster.save_model(discriminator, os.path.join(save_dir, "discriminator"), shard=True)
                            booster.save_optimizer(optimizer, os.path.join(save_dir, "optimizer"), shard=True, size_per_shard=4096)
                            booster.save_optimizer(disc_optimizer, os.path.join(save_dir, "disc_optimizer"), shard=True, size_per_shard=4096)

                            if lr_scheduler is not None:
                                booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, "lr_scheduler"))
                            if disc_lr_scheduler is not None:
                                booster.save_lr_scheduler(disc_lr_scheduler, os.path.join(save_dir, "disc_lr_scheduler"))

                            running_states = {
                                "epoch": epoch,
                                "step": step+1,
                                "global_step": global_step+1,
                                "sample_start_index": (step+1) * cfg.batch_size,
                            }
                            if coordinator.is_master():
                                save_json(running_states, os.path.join(save_dir, "running_states.json"))
                            dist.barrier()
                            logger.info(
                                f"Saved checkpoint at epoch {epoch} step {step + 1} global_step {global_step + 1} to {exp_dir}"
                            )

                        # p.step()

                    # print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
                    

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        dataloader.sampler.set_start_index(0)
        start_step = 0

if __name__ == "__main__":
    main()
