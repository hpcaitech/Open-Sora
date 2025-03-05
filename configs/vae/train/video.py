# Define dataset
dataset = dict(
    type="video_text",
    transform_name="resize_crop",
    fps_max=16,
)
bucket_config = {
    "256px_ar1:1": {32: (1.0, 1)},
}

grad_checkpoint = False
num_bucket_build_workers = 64
num_workers = 12
prefetch_factor = 2
pin_memory_cache_pre_alloc_numels = [50 * 1024 * 1024] * num_workers * prefetch_factor

# Define model
model = dict(
    type="autoencoder_3d",
    from_pretrained="/mnt/jfs-hdd/sora/checkpoints/pretrained_models/flux/FLUX.1-dev_convert/ae_central_inflate_zero_init.safetensors",
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    z_channels=16,
    scale_factor=1.0,
    shift_factor=0.0,
)
# discriminator = dict(type="N_Layer_discriminator_3D", from_pretrained=None, input_nc=3, n_layers=5, conv_cls="conv3d")

# == loss weights ==
mixed_strategy = "mixed_video_image"
mixed_image_ratio = 0.25  # 1:4

opl_loss_weight = 1e5
# opl_loss_weight = 1e3

vae_loss_config = dict(
    perceptual_loss_weight=1.0,
    kl_loss_weight=5e-4,
    # kl_loss_weight=5e-6,
    logvar_init=0.0,
)  # reconstruction loss (nll) + lpips similarity loss + kl loss

gen_loss_config = dict(
    gen_start=0,
    disc_factor=1e5,
    disc_weight=0.5,
)

disc_loss_config = dict(
    disc_start=0,
    disc_factor=1.0,
    disc_loss_type="wgan-gp",
)

# == optimizer ==
optim = dict(
    cls="HybridAdam",
    lr=5e-6,
    eps=1e-8,
    weight_decay=0.0,
    adamw_mode=True,
    betas=(0.9, 0.98),
)
optim_discriminator = dict(
    cls="HybridAdam",
    lr=1e-5,
    eps=1e-8,
    weight_decay=0.0,
    adamw_mode=True,
    betas=(0.9, 0.98),
)

lr_scheduler = dict(warmup_steps=0)
disc_lr_scheduler = dict(warmup_steps=0)

update_warmup_steps = True
# start_epoch = start_step = 0

grad_clip = 1.0

# Acceleration settings
dtype = "bf16"
plugin = "zero2"
plugin_config = dict(
    reduce_bucket_size_in_m=128,
    overlap_allgather=False,
)

# Others
seed = 42
outputs = "outputs"
epochs = 100
log_every = 10
ckpt_every = 2000
keep_n_latest = 50
ema_decay = 0.99

# wandb
wandb_project = "mmdit_vae"
