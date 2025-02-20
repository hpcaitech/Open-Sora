# Dataset settings
dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop",
)
bucket_config = {
    "360p": {1: (1.0, 8)},
    # ---
    "480p": {1: (0.3, 4)},
    # ---
    "720p": {1: (0.2, 2)},
    # ---
    "1080p": {1: (1.0, 1)},
    # ---
    "2048": {1: (0.7, 1)},
}
grad_checkpoint = False
warmup = 1000

# Acceleration settings
num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
plugin = "zero2"

# Model settings
model = dict(
    type="STDiT3-XL/2",
    from_pretrained="outputs/0373-STDiT3-XL-2/epoch3-global_step36000/ema.pt",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    adapt_16ch=True,
    skip_temporal=True,
)
# vae = dict(
#     type="VideoAutoencoderKL",
#     from_pretrained="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
#     subfolder="vae",
#     scaling_factor=0.13025,
# )
vae = dict(
    type="OpenSoraVAE_V1_3",
    from_pretrained="/home/guoxinying/open_source_video_ocean_V1/OpenSora-VAE-v1.3",
    z_channels=16,
    micro_batch_size=1,
    micro_batch_size_2d=4,
    micro_frame_size=17,
    use_tiled_conv3d=True,
    tile_size=4,
    force_huggingface=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
    shardformer=True,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    sample_method="logit-normal",
)

# Log settings
seed = 42
outputs = "outputs"
wandb = False
epochs = 1000
log_every = 10
ckpt_every = 500

# optimization settings
load = None
grad_clip = 1.0
lr = 1e-4
adam_eps = 1e-8
ema_decay = 0.99
