_base_ = ["video.py"]

dtype = "bf16"
ckpt_every = 2500

mixed_image_ratio = 0.01
restart_disc = True

# == loss weights ==
opl_loss_weight = 0.0
vae_loss_config = dict(
    kl_loss_weight=1e-6,
    perceptual_loss_weight=1.0,
)
gen_loss_config = dict(
    disc_factor=1.0,
    disc_weight=0.5,  # proportion of grad norm w.r.t. nll loss
)
disc_loss_config = dict(
    disc_loss_type="hinge",
)

# == optimizer ==
optim = dict(
    lr=1e-6,
    betas=(0.9, 0.999),
)
optim_discriminator = dict(
    lr=1e-6,
    betas=(0.9, 0.999),
)
ema_decay = None

# TORCH_COMPILE_DISABLE=1 torchrun --nproc_per_node 8 scripts/vae/train.py configs/vae/train/256px.py --dataset.data-path cache/meta/tmp_vae_bpp_bppmin-0.035.csv --model.from_pretrained /mnt/jfs-hdd/sora/checkpoints/pretrained_models/vae_videoocean_1025.pt --wandb True

# TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=7 torchrun --master-port 14312 --nproc_per_node 1 scripts/vae/inference.py configs/vae/inference/video.py --dataset.data-path /mnt/jfs-hdd/sora/data/eval_loss/eval_vid.csv --save-dir samples/vae_12_14 --ckpt-path outputs/241214_012637-vae_train_video/epoch0-global_step17000 --eval-setting 32x512x512 --type video

# CUDA_VISIBELE_DEVICES=6 python vae/eval_common_metric.py --batch_size 1 --real_video_dir ~/Video-Ocean/samples/vae_12_13/orig --generated_video_dir ~/Video-Ocean/samples/vae_12_13/recn --device cuda --sample_fps 16 --crop_size 512 --resolution 512 --num_frames 32 --sample_rate 1 --metric ssim psnr lpips
