_base_ = ["video.py"]

bucket_config = {
    "_delete_": True,
    "512px_ar1:1": {32: (1.0, 1)},
}
grad_checkpoint = True

# == loss weights ==
opl_loss_weight = 1e3
vae_loss_config = dict(
    kl_loss_weight=5e-4,
    perceptual_loss_weight=0.1,
)
gen_loss_config = dict(
    disc_factor=1.0,
    disc_weight=0.2,
)
optim = dict(
    cls="HybridAdam",
    lr=5e-6,
    eps=1e-8,
    weight_decay=0.0,
    adamw_mode=True,
    betas=(0.9, 0.999),
)

# TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=7 torchrun --master-port 14312 --nproc_per_node 1 scripts/vae/inference.py configs/vae/inference/video.py --dataset.data-path /mnt/jfs-hdd/sora/data/eval_loss/eval_vid.csv --save-dir samples/vae_12_14 --ckpt-path outputs/241214_012637-vae_train_video/epoch0-global_step17000 --eval-setting 32x512x512 --type video

# CUDA_VISIBELE_DEVICES=6 python vae/eval_common_metric.py --batch_size 1 --real_video_dir ~/Video-Ocean/samples/vae_12_13/orig --generated_video_dir ~/Video-Ocean/samples/vae_12_13/recn --device cuda --sample_fps 16 --crop_size 512 --resolution 512 --num_frames 32 --sample_rate 1 --metric ssim psnr lpips

# TORCH_COMPILE_DISABLE=1 torchrun --nproc_per_node 8 --master_port 30303 scripts/vae/train.py configs/vae/train/video.py --dataset.data-path /mnt/ddn/sora/meta/merge/vo1.1_stage-1_res-256/20241210.csv --model.from_pretrained /mnt/jfs-hdd/sora/checkpoints/pretrained_models/vae_videoocean_1025.pt

# TORCH_COMPILE_DISABLE=1 torchrun --nproc_per_node 8 --master_port 30303 scripts/vae/train.py configs/vae/train/512px.py --dataset.data-path /mnt/ddn/sora/meta/merge/vo1.1_stage-1_res-256/20241210.csv --model.from_pretrained /mnt/jfs-hdd/sora/checkpoints/pretrained_models/vae_videoocean_1025.pt
