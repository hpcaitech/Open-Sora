# Dataset settings
dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop",
)
bucket_config = {
    "720p": {
        1: (1.0, 30),
        49: (1.0, 2),  # 15
        65: (1.0, 1),  # 20
        81: (1.0, 1),  # 25
        97: (1.0, 1),  # 30
        113: (1.0, 1),  # 35
    },
}
grad_checkpoint = True

# Acceleration settings
num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
plugin = "zero1"

# Model settings
model = dict(
    type="STDiT3-XL/2",
    from_pretrained=None,
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    kernel_size=(8, 8, -1),  # H W T
    use_spatial_rope=True,
)
vae = dict(
    type="OpenSoraVAE_V1_3",
    from_pretrained="/home/guoxinying/open_source_video_ocean_V1/OpenSora-VAE-v1.3",
    z_channels=16,
    micro_batch_size=1,
    micro_batch_size_2d=4,
    micro_frame_size=17,
    use_tiled_conv3d=True,
    tile_size=4,
    normalization="video",
    temporal_overlap=True,
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
    sample_method="logit-normal",
    use_timestep_transform=True,
)

# Log settings
seed = 42
outputs = "outputs"
wandb = False
epochs = 1000
log_every = 10
ckpt_every = 250

# optimization settings
lr = 5e-5
warmup_steps = 1000
grad_clip = 1.0
adam_eps = 1e-15
ema_decay = 0.99
accumulation_steps = 2
