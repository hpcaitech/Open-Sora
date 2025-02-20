# Dataset settings
dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop",
)

bucket_config = {
    "360p": {
        49: (1.0, 10),  # 15
        65: (1.0, 7),  # 20
        81: (1.0, 6),  # 25
        97: (1.0, 5),  # 30
        113: (1.0, 4),  # 35
    },
    "720p": {
        49: (0.25, 2),  # 15
        65: (0.25, 1),  # 20
        81: (0.25, 1),  # 25
        97: (0.25, 1),  # 30
        113: (0.25, 1),  # 35
    },
}

# i2v & v2v condition
drop_condition = {
    "cond": 0.05,
    "text": 0.05,
    "null": 0.05,
    "keep": 0.85,  # 85% of the time don't drop anything
}

# i2v & v2v condition
mask_types = {
    "i2v_head": 5,
    "i2v_tail": 2,
    "i2v_loop": 2,
    "v2v_head": 1,
    "v2v_head_noisy": 2,
    "v2v_tail": 1,
    "other": 1,
    "none": 2,
}

grad_checkpoint = True

# Acceleration settings
num_workers = 8  # 8
num_bucket_build_workers = 16  # 16
dtype = "bf16"
plugin = "zero2"

# Model settings
model = dict(
    type="STDiT3-XL/2",
    from_pretrained=None,
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    freeze_y_embedder=True,
    class_dropout_prob=0.1,  # i2v & v2v change
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
    from_pretrained="pretrained_models/t5-v1_1-xxl",
    model_max_length=300,
    shardformer=True,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    sample_method="logit-normal",
    drop_condition=drop_condition,  # i2v & v2v random drop text/image/video condition
)

# Log settings
seed = 4207
outputs = "outputs/"
wandb = False
epochs = 1000
log_every = 8
ckpt_every = 200

# optimization settings
grad_clip = 1.0
lr = 1e-4
ema_decay = 0.99
adam_eps = 1e-15
warmup_steps = 1000

# v2v params
accumulation_steps = 1
