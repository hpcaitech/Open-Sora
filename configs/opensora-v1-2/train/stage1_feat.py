# Dataset settings
dataset = dict(type="BatchFeatureDataset")
grad_checkpoint = True
num_workers = 4

# Acceleration settings
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
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    sample_method="logit-normal",
)

vae_out_channels = 4
model_max_length = 300
text_encoder_output_dim = 4096
load_video_features = True
load_text_features = True

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
