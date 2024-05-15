# Dataset settings
dataset = dict(
    type="VideoTextDataset",
    data_path=None,
    num_frames=16,
    frame_interval=3,
    image_size=(256, 256),
    transform_name="resize_crop",
) # just occupy the space.... actually in evaluation we will create dataset for different resolutions
eval_config = {  # 2s/it
    "144p": {1: (0.5, 40), 34: (1.0, 10), 51: (1.0, 10), 102: (1.0, 5), 204: (1.0, 2)},
    # # ---
    "256": {1: (0.6, 40), 34: (0.5, 10), 51: (0.5, 5), 68: (0.5, 5), 136: (0.0, 4)},
    "240p": {1: (0.6, 40), 34: (0.5, 10), 51: (0.5, 5), 68: (0.5, 5), 136: (0.0, 4)},
    # # ---
    "360p": {1: (0.5, 20), 34: (0.2, 8), 102: (0.0, 4)},
    "512": {1: (0.5, 10), 34: (0.2, 8), 102: (0.0, 4)},
    # ---
    "480p": {1: (0.2, 10), 17: (0.3, 5), 68: (0.0, 2)},,
    # ---
    "720p": {1: (0.1, 5)},
    "1024": {1: (0.1, 4)},
    # ---
    "1080p": {1: (0.1, 2)},
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
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderPipeline",
    from_pretrained="/home/zhengzangwei/projs/Open-Sora-dev/pretrained_models/vae-v3",
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
    shardformer=False,
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
outputs = "outputs/eval_loss"
epochs = 1000
log_every = 10
ckpt_every = 500

# optimization settings
load = None
grad_clip = 1.0
lr = 1e-4
ema_decay = 0.99
adam_eps = 1e-15

# eval
num_eval_samples = 40 # num eval samples per (res, num_frames, ar, t)
num_eval_timesteps = 20
