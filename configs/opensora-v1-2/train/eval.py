# Dataset settings
dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop",
    frame_interval=1,
)

bucket_config = {  # 20s/it
    "1024": {1: (1.0, 1)},
}

grad_checkpoint = True
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
    from_pretrained="pretrained_models/vae-pipeline",
    micro_frame_size=17,
    shift=(-0.10, 0.34, 0.27, 0.98),
    scale=(3.85, 2.32, 2.33, 3.06),
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
    use_timestep_transform=False,
    # sample_method="logit-normal",
)

# Mask settings
# mask_ratios = {
#     "random": 0.1,
#     "intepolate": 0.01,
#     "quarter_random": 0.01,
#     "quarter_head": 0.01,
#     "quarter_tail": 0.01,
#     "quarter_head_tail": 0.01,
#     "image_random": 0.05,
#     "image_head": 0.1,
#     "image_tail": 0.05,
#     "image_head_tail": 0.05,
# }

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
lr = 2e-4
ema_decay = 0.99
adam_eps = 1e-15
