import os

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype
from opensora.datasets import prepare_dataloader, prepare_variable_dataloader
from opensora.datasets import DATASETS, MODELS, build_module
from opensora.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)
from tqdm import tqdm
from opensora.models.vae.vae_3d_v2 import VEALoss, DiscriminatorLoss, AdversarialLoss, pad_at_dim

from einops import rearrange
from colossalai.utils import get_current_device


def main():
    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    cfg = parse_configs(training=False)
    print(cfg)

    # init distributed
    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()

    # if coordinator.world_size > 1:
    #     set_sequence_parallel_group(dist.group.WORLD)
    #     enable_sequence_parallelism = True
    # else:
    #     enable_sequence_parallelism = False

    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    device = get_current_device()
    dtype = to_torch_dtype(cfg.dtype)




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
    print(f"Dataset contains {len(dataset):,} videos ({cfg.data_path})")

    total_batch_size = cfg.batch_size * dist.get_world_size() // cfg.sp_size
    print(f"Total batch size: {total_batch_size}")

    # ======================================================
    # 4. build model & load weights
    # ======================================================
    # 3.1. build model
    if cfg.get("use_pipeline") == True:
        # use 2D VAE, then temporal VAE
        vae_2d = build_module(cfg.vae_2d, MODELS)
    vae = build_module(cfg.model, MODELS, device=device)
    discriminator = build_module(cfg.discriminator, MODELS, device=device)

    # 3.2. move to device & eval
    if cfg.get("use_pipeline") == True:
        vae_2d.to(device, dtype).eval()
    vae = vae.to(device, dtype).eval()
    discriminator = discriminator.to(device, dtype).eval()

    # 3.4. support for multi-resolution
    model_args = dict()
    if cfg.multi_resolution:
        image_size = cfg.dataset.image_size
        hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        model_args["data_info"] = dict(ar=ar, hw=hw)

    # ======================================================
    # 4. inference
    # ======================================================
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)



    # 4.1. batch generation
    
    # define loss function
    if cfg.calc_loss:
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
    
    
    running_loss = 0.0
    running_disc_loss = 0.0
    loss_steps = 0

    disc_time_downsample_factor = 2 ** len(cfg.discriminator.channel_multipliers)
    if cfg.datasets.num_frames % disc_time_downsample_factor != 0:
        disc_time_padding = disc_time_downsample_factor - cfg.datasets.num_frames % disc_time_downsample_factor
    else:
        disc_time_padding = 0
    video_contains_first_frame = cfg.video_contains_first_frame

    lecam_ema_real = torch.tensor(0.0)
    lecam_ema_fake = torch.tensor(0.0)

    total_steps = len(dataloader)
    if cfg.max_test_samples > 0:
        total_steps = min(int(cfg.max_test_samples//cfg.batch_size), total_steps)
        print(f"limiting test dataset to {int(cfg.max_test_samples//cfg.batch_size) * cfg.batch_size}")
    dataloader_iter = iter(dataloader)

    with tqdm(
        range(total_steps),
        # desc=f"Avg Loss: {running_loss}",
        disable=not coordinator.is_master(),
        total=total_steps,
        initial=0,
    ) as pbar:
        for step in pbar:
            batch = next(dataloader_iter)
            x = batch["video"].to(device, dtype)  # [B, C, T, H, W]

            is_image = x.ndim == 4
            if is_image:
                video = rearrange(x, 'b c ... -> b c 1 ...')
                video_contains_first_frame = True
            else:
                video = x

            #  ===== Spatial VAE =====
            if cfg.get("use_pipeline") == True:
                with torch.no_grad():
                    video_enc_spatial = vae_2d.encode(video)

                recon_video, posterior = vae(
                    video_enc_spatial,
                    video_contains_first_frame = video_contains_first_frame
                )
            else:
                recon_video, posterior = vae(
                    video,
                    video_contains_first_frame = video_contains_first_frame
                )

            if cfg.calc_loss:
                #  ====== Calc Loss ======
                # simple nll loss
                nll_loss, weighted_nll_loss, weighted_kl_loss = vae_loss_fn(
                    video_enc_spatial,
                    recon_video,
                    posterior,
                    split = "eval"
                )

                fake_video = pad_at_dim(recon_video, (disc_time_padding, 0), value = 0., dim = 2)
                fake_logits = discriminator(fake_video.contiguous()) # TODO: take out contiguous?
                adversarial_loss = adversarial_loss_fn(
                    fake_logits,
                    nll_loss, 
                    vae.get_last_layer(),
                    cfg.discriminator_start+1, # Hack to use discriminator
                    is_training = vae.training,
                )
                
                vae_loss = weighted_nll_loss + weighted_kl_loss + adversarial_loss
                
                #  ====== Discriminator Loss ======
                real_video = pad_at_dim(video_enc_spatial, (disc_time_padding, 0), value = 0., dim = 2)
                fake_video = pad_at_dim(recon_video, (disc_time_padding, 0), value = 0., dim = 2)

                if cfg.gradient_penalty_loss_weight is not None and cfg.gradient_penalty_loss_weight > 0.0:
                    real_video = real_video.requires_grad_()
                    real_logits = discriminator(real_video.contiguous()) # SCH: not detached for now for gradient_penalty calculation
                else:
                    real_logits = discriminator(real_video.contiguous().detach()) 

                fake_logits = discriminator(fake_video.contiguous().detach())
                weighted_d_adversarial_loss, lecam_loss, gradient_penalty_loss = disc_loss_fn(
                    real_logits, 
                    fake_logits, 
                    cfg.discriminator_start+1, # Hack to use discriminator 
                    lecam_ema_real = lecam_ema_real, 
                    lecam_ema_fake = lecam_ema_fake, 
                    real_video = real_video if cfg.gradient_penalty_loss_weight is not None else None,
                )

                disc_loss = weighted_d_adversarial_loss + lecam_loss + gradient_penalty_loss

                if cfg.ema_decay is not None: 
                    # SCH: TODO: is this written properly like this for moving average? e.g. distributed training etc.
                    lecam_ema_real = lecam_ema_real * cfg.ema_decay + (1 - cfg.ema_decay) * torch.mean(real_logits.clone().detach())
                    lecam_ema_fake = lecam_ema_fake * cfg.ema_decay + (1 - cfg.ema_decay) * torch.mean(fake_logits.clone().detach())


                loss_steps += 1
                running_disc_loss = disc_loss.item()/loss_steps + disc_loss * ((loss_steps - 1) / loss_steps)
                running_loss = vae_loss.item()/ loss_steps + running_loss * ((loss_steps - 1) / loss_steps)


            #  ===== Spatial VAE =====


            if coordinator.is_master():
                if cfg.get("use_pipeline") == True:
                    with torch.no_grad(): # 2nd stage decoding
                        recon_pipeline = vae_2d.decode(recon_video)
                        recon_2d = vae_2d.decode(video_enc_spatial)

                    for idx, (sample_original, sample_pipeline, sample_2d) in enumerate(zip(video, recon_pipeline, recon_2d)):
                        pos = step * cfg.batch_size + idx
                        save_path = os.path.join(save_dir, f"sample_{pos}")
                        save_sample(sample_original, fps=cfg.fps, save_path=save_path+"_original")
                        save_sample(sample_2d, fps=cfg.fps, save_path=save_path+"_2d")
                        save_sample(sample_pipeline, fps=cfg.fps, save_path=save_path+"_pipeline")

                else:
                    for idx, (original, recon) in enumerate(zip(video, recon_video)):
                        pos = step * cfg.batch_size + idx
                        save_path = os.path.join(save_dir, f"sample_{pos}")
                        save_sample(original, fps=cfg.fps, save_path=save_path+"_original")
                        save_sample(recon, fps=cfg.fps, save_path=save_path+"_recon")



    if cfg.calc_loss:
        print("test vae loss:", running_loss)
        print("test disc loss:", running_disc_loss)

    

if __name__ == "__main__":
    main()
