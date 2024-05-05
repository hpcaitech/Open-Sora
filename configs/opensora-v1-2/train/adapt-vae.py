# Dataset settings
dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop",
)
bucket_config = {
    "144p": {1: (1.0, 64)},
    # ---
    "256": {1: (0.5, 48)},
    "240p": {1: (0.5, 48), 16: (1.0, 8)},
    # ---
    "360p": {1: (0.5, 18)},
    "512": {1: (0.5, 18)},
    # ---
    "480p": {1: (0.5, 8)},
    # ---
    "720p": {1: (0.2, 4)},
    "1024": {1: (0.2, 4)},
    # ---
    "1080p": {1: (0.3, 2)},
    # ---
    "2048": {1: (0.3, 1)},
}
grad_checkpoint = False

# Acceleration settings
num_workers = 4
num_bucket_build_workers = 16
dtype = "bf16"
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
    type="VideoAutoencoderPipeline",
    from_pretrained="pretrained_models/vae-v2",
    micro_frame_size=16,
    vae_2d=dict(
        type="VideoAutoencoderKL",
        from_pretrained="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
        subfolder="vae",
        micro_batch_size=4,
        local_files_only=True,
    ),
    vae_temporal=dict(
        type="VAE_Temporal_SD",
        from_pretrained=None,
    ),
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
lr = 1e-4
ema_decay = 0.99
adam_eps = 1e-15
