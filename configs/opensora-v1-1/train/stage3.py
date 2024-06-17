# Define dataset
dataset = dict(
    type="VariableVideoTextDataset",
    data_path=None,
    num_frames=None,
    frame_interval=3,
    image_size=(None, None),
    transform_name="resize_crop",
)
bucket_config = {  # 13s/it
    "144p": {1: (1.0, 200), 16: (1.0, 36), 32: (1.0, 18), 64: (1.0, 9), 128: (1.0, 4)},
    "256": {1: (0.8, 200), 16: (0.5, 22), 32: (0.5, 11), 64: (0.5, 6), 128: (0.8, 4)},
    "240p": {1: (0.8, 200), 16: (0.5, 22), 32: (0.5, 10), 64: (0.5, 6), 128: (0.5, 3)},
    "360p": {1: (0.5, 120), 16: (0.5, 9), 32: (0.5, 4), 64: (0.5, 2), 128: (0.5, 1)},
    "512": {1: (0.5, 120), 16: (0.5, 9), 32: (0.5, 4), 64: (0.5, 2), 128: (0.8, 1)},
    "480p": {1: (0.4, 80), 16: (0.6, 6), 32: (0.6, 3), 64: (0.6, 1), 128: (0.0, None)},
    "720p": {1: (0.4, 40), 16: (0.6, 3), 32: (0.6, 1), 96: (0.0, None)},
    "1024": {1: (0.3, 40)},
}
mask_ratios = {
    "identity": 0.75,
    "quarter_random": 0.025,
    "quarter_head": 0.025,
    "quarter_tail": 0.025,
    "quarter_head_tail": 0.05,
    "image_random": 0.025,
    "image_head": 0.025,
    "image_tail": 0.025,
    "image_head_tail": 0.05,
}

# Define acceleration
num_workers = 8
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

batch_size = None
lr = 2e-5
grad_clip = 1.0
