num_workers = 8
dtype = "bf16"
seed = 42
num_eval_timesteps = 10

# Dataset settings
dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop",
)

# just occupy the space.... actually in evaluation we will create dataset for different resolutions
bucket_config = {  # 20s/it
    "144p": {1: (1.0, 100), 51: (1.0, 30), 102: ((1.0, 0.33), 20), 204: ((1.0, 0.1), 8), 408: ((1.0, 0.1), 4)},
    # ---
    "240p": {1: (0.3, 100), 51: (0.4, 24), 102: ((0.4, 0.33), 12), 204: ((0.4, 0.1), 4), 408: ((0.4, 0.1), 2)},
    # ---
    "360p": {1: (0.2, 60), 51: (0.15, 12), 102: ((0.15, 0.33), 6), 204: ((0.15, 0.1), 2), 408: ((0.15, 0.1), 1)},
    # ---
    "480p": {1: (0.1, 40), 51: (0.3, 6), 102: (0.3, 3), 204: (0.3, 1), 408: (0.0, None)},
    # ---
    "720p": {1: (0.05, 20), 51: (0.3, 2), 102: (0.3, 1), 204: (0.0, None)},
    # ---
    "1080p": {1: (0.1, 10)},
    # ---
    "2048": {1: (0.1, 5)},
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
    from_pretrained="pretrained_models/vae-pipeline",
    micro_frame_size=17,
    micro_batch_size=4,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
    local_files_only=True,
)
scheduler = dict(type="rflow")
