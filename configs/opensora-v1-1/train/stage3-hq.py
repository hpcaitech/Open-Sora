# Define dataset
dataset = dict(
    type="VariableVideoTextDataset",
    data_path=None,
    num_frames=None,
    frame_interval=3,
    image_size=(None, None),
    transform_name="resize_crop",
)
bucket_config = {  # 18s/it
    "144p": {1: (1.0, 500), 16: (1.0, 36), 32: (1.0, 18), 64: (1.0, 9), 128: (1.0, 4)},
    "256": {1: (0.8, 300), 16: (0.5, 24), 32: (0.5, 12), 64: (0.5, 7), 128: (0.8, 4)},
    "240p": {1: (0.8, 300), 16: (0.5, 24), 32: (0.5, 12), 64: (0.5, 7), 128: (0.5, 4)},
    "360p": {1: (0.5, 150), 16: (0.5, 12), 32: (0.5, 6), 64: (0.5, 3), 128: (0.5, 1)},
    "512": {1: (0.5, 150), 16: (0.5, 12), 32: (0.5, 6), 64: (0.5, 3), 128: (0.8, 1)},
    "480p": {1: (0.4, 100), 16: (0.6, 8), 32: (0.6, 4), 64: (0.6, 2), 128: (0.8, 1)},
    "720p": {1: (0.4, 50), 16: (0.6, 4), 32: (0.6, 2), 64: (0.6, 1), 96: (0.0, None)},
    "1024": {1: (0.3, 50)},
    "1080p": {1: (0.3, 20)},
}
mask_ratios = {
    "mask_no": 0.75,
    "mask_quarter_random": 0.025,
    "mask_quarter_head": 0.025,
    "mask_quarter_tail": 0.025,
    "mask_quarter_head_tail": 0.05,
    "mask_image_random": 0.025,
    "mask_image_head": 0.025,
    "mask_image_tail": 0.025,
    "mask_image_head_tail": 0.05,
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
    enable_flashattn=True,
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
