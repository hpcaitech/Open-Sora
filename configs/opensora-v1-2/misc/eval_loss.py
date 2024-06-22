num_workers = 8
dtype = "bf16"
seed = 42
num_eval_timesteps = 10

# Dataset settings
dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop",
)

bucket_config = {
    "144p": {1: (None, 100), 51: (None, 30), 102: (None, 20), 204: (None, 8), 408: (None, 4)},
    # ---
    "240p": {1: (None, 100), 51: (None, 24), 102: (None, 12), 204: (None, 4), 408: (None, 2)},
    # ---
    "360p": {1: (None, 60), 51: (None, 12), 102: (None, 6), 204: (None, 2), 408: (None, 1)},
    # ---
    "480p": {1: (None, 40), 51: (None, 6), 102: (None, 3), 204: (None, 1)},
    # ---
    "720p": {1: (None, 20), 51: (None, 2), 102: (None, 1)},
    # ---
    "1080p": {1: (None, 10)},
    # ---
    "2048": {1: (None, 5)},
}

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
    local_files_only=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
    local_files_only=True,
)
scheduler = dict(type="rflow")
