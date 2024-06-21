# Dataset settings
dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop",
)

# == Config 1: Webvid ==
# base: (512, 408), 12s/it
grad_checkpoint = True
base = ("512", "408")
base_step_time = 12
bucket_config = {
    "144p": {
        1: (475, 0),
        51: (51, 0),
        102: (27, 0),
        204: (13, 0),
        408: (6, 0),
    },
    # ---
    "240p": {
        1: (297, 200),  # 8.25
        51: (20, 0),
        102: (10, 0),
        204: (5, 0),
        408: (2, 0),
    },
    # ---
    "512": {
        1: (141, 0),
        51: (8, 0),
        102: (4, 0),
        204: (2, 0),
        408: (1, 0),
    },
    # ---
    "480p": {
        1: (89, 0),
        51: (5, 0),
        102: (2, 0),
        204: (1, 0),
    },
    # ---
    "1024": {
        1: (36, 0),
        51: (1, 0),
    },
    # ---
    "1080p": {1: (5, 0)},
    # ---
    "2048": {1: (5, 0)},
}

# == Config 1 ==
# base: (512, 408), 16s/it

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
    type="OpenSoraVAE_V1_2",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
    micro_frame_size=17,
    micro_batch_size=4,
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
    use_timestep_transform=True,
    sample_method="logit-normal",
)

# Mask settings
mask_ratios = {
    "random": 0.2,
    "intepolate": 0.01,
    "quarter_random": 0.01,
    "quarter_head": 0.01,
    "quarter_tail": 0.01,
    "quarter_head_tail": 0.01,
    "image_random": 0.05,
    "image_head": 0.1,
    "image_tail": 0.05,
    "image_head_tail": 0.05,
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
lr = 2e-4
ema_decay = 0.99
adam_eps = 1e-15
