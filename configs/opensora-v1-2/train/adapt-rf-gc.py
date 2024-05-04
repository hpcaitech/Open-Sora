# Dataset settings
dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop",
)
bucket_config = {
    "144p": {1: (1.0, 200)},
    # ---
    "256": {1: (0.5, 200)},
    "240p": {1: (0.5, 200)},
    # ---
    "360p": {1: (0.5, 120)},
    "512": {1: (0.5, 120)},
    # ---
    "480p": {1: (0.5, 80)},
    # ---
    "720p": {1: (0.2, 40)},
    "1024": {1: (0.1, 40)},
    # ---
    "1080p": {1: (0.1, 20)},
    # ---
    "2048": {1: (0.1, 10)},
}

# Acceleration settings
num_workers = 4
num_bucket_build_workers = 16
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Model settings
model = dict(
    type="STDiT3-XL/2",
    from_pretrained=None,
    qk_norm=True,
    enable_flashattn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    micro_batch_size=4,
    from_pretrained="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
    subfolder="vae",
    local_files_only=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
    shardformer=True,
    local_files_only=True,
)
scheduler = dict(
    type="rflow",
    use_discrete_timesteps=False,
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
batch_size = None
grad_clip = 1.0
# lr = 1e-4
lr = 2e-5
ema_decay = 0.99
adam_eps = 1e-15
