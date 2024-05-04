# Define dataset
dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop",
)
bucket_config = {
    "144p": {1: (1.0, 64)},
    # ---
    "256": {1: (0.5, 48)},
    "240p": {1: (0.5, 48)},
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
    "2048": {1: (1.0, 1)},
}

# Define acceleration
num_workers = 4
num_bucket_build_workers = 16
dtype = "bf16"
grad_checkpoint = False
plugin = "zero2"
sp_size = 1

# Define model
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
    type="iddpm",
    timestep_respacing="",
)

# Others
seed = 42
outputs = "outputs"
wandb = False

epochs = 1000
log_every = 10
ckpt_every = 500
load = None

batch_size = 10  # only for logging
lr = 2e-5
grad_clip = 1.0