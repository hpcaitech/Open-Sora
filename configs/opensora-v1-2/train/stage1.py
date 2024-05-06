# Dataset settings
dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop",
    frame_interval=1,
)
bucket_config = {  # 2s/it
    "144p": {1: (0.5, 48), 34: (1.0, 2), 51: (1.0, 4), 102: (1.0, 2), 204: (1.0, 1)},
    # ---
    "256": {1: (0.6, 20), 34: (0.5, 2), 51: (0.5, 1), 68: (0.5, 1), 136: (0.0, None)},
    "240p": {1: (0.6, 20), 34: (0.5, 2), 51: (0.5, 1), 68: (0.5, 1), 136: (0.0, None)},
    # ---
    "360p": {1: (0.5, 8), 34: (0.2, 1), 102: (0.0, None)},
    "512": {1: (0.5, 8), 34: (0.2, 1), 102: (0.0, None)},
    # ---
    "480p": {1: (0.2, 4), 17: (0.3, 1), 68: (0.0, None)},
    # ---
    "720p": {1: (0.1, 2)},
    "1024": {1: (0.1, 2)},
    # ---
    "1080p": {1: (0.1, 1)},
}
grad_checkpoint = False  # determine batch size
batch_size = None

# Acceleration settings
num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
plugin = "zero2"

# Model settings
model = dict(
    type="STDiT3-XL/2",
    from_pretrained=None,
    qk_norm=True,
    enable_flashattn=True,
    enable_layernorm_kernel=True,
    only_train_temporal=True,
)
vae = dict(
    type="VideoAutoencoderPipeline",
    from_pretrained="pretrained_models/vae-v2",
    micro_frame_size=17,
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

# Mask settings
mask_ratios = {
    "mask_random": 0.4,
    "mask_intepolate": 0.01,
    "mask_quarter_random": 0.01,
    "mask_quarter_head": 0.01,
    "mask_quarter_tail": 0.01,
    "mask_quarter_head_tail": 0.01,
    "mask_image_random": 0.01,
    "mask_image_head": 0.01,
    "mask_image_tail": 0.01,
    "mask_image_head_tail": 0.01,
}

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
ema_decay = 0.99
adam_eps = 1e-15
