# Dataset settings
dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop",
    dummy_text_feature=True,
)

# webvid
bucket_config = {  # 12s/it
    "144p": {1: (1.0, 475), 51: (1.0, 51), 102: ((1.0, 0.33), 27), 204: ((1.0, 0.1), 13), 408: ((1.0, 0.1), 6)},
    # ---
    "256": {1: (0.4, 297), 51: (0.5, 20), 102: ((0.5, 0.33), 10), 204: ((0.5, 0.1), 5), 408: ((0.5, 0.1), 2)},
    "240p": {1: (0.3, 297), 51: (0.4, 20), 102: ((0.4, 0.33), 10), 204: ((0.4, 0.1), 5), 408: ((0.4, 0.1), 2)},
    # ---
    "360p": {1: (0.2, 141), 51: (0.15, 8), 102: ((0.15, 0.33), 4), 204: ((0.15, 0.1), 2), 408: ((0.15, 0.1), 1)},
    "512": {1: (0.1, 141)},
    # ---
    "480p": {1: (0.1, 89)},
    # ---
    "720p": {1: (0.05, 36)},
    "1024": {1: (0.05, 36)},
    # ---
    "1080p": {1: (0.1, 5)},
    # ---
    "2048": {1: (0.1, 5)},
}

grad_checkpoint = True

load_text_features = True

# Acceleration settings
num_workers = 0
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
    freeze_y_embedder=True,
    skip_y_embedder=True,
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
ckpt_every = 1

# optimization settings
load = None
grad_clip = 1.0
lr = 2e-4
ema_decay = 0.99
adam_eps = 1e-15
