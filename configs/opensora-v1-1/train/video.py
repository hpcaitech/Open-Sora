# Define dataset
dataset = dict(
    type="VariableVideoTextDataset",
    data_path=None,
    num_frames=None,
    frame_interval=3,
    image_size=(None, None),
    transform_name="resize_crop",
)
bucket_config = {  # 6s/it
    "240p": {16: (1.0, 16), 32: (1.0, 8), 64: (1.0, 4), 128: (1.0, 2)},
    "256": {1: (1.0, 256)},
    "512": {1: (0.5, 80)},
    "480p": {1: (0.4, 52), 16: (0.4, 4), 32: (0.0, None)},
    "720p": {16: (0.1, 2), 32: (0.0, None)},  # No examples now
    "1024": {1: (0.3, 20)},
    "1080p": {1: (0.3, 8)},
}

# Define acceleration
num_workers = 4
num_bucket_build_workers = 16
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="STDiT2-XL/2",
    from_pretrained=None,
    input_sq_size=512,  # pretrained model is trained on 512x512
    qk_norm=True,
    qk_norm_legacy=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    micro_batch_size=4,
    local_files_only=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=200,
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
