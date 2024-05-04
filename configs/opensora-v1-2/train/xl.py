# Define dataset
dataset = dict(
    type="VariableVideoTextDataset",
    frame_interval=3,
    transform_name="resize_crop",
)
bucket_config = {
    "240p": {16: (1.0, 2)},
}
mask_ratios = {  # 20%
    "mask_random": 0.10,
    "mask_quarter_random": 0.01,
    "mask_quarter_head": 0.01,
    "mask_quarter_tail": 0.01,
    "mask_quarter_head_tail": 0.01,
    "mask_image_random": 0.01,
    "mask_image_head": 0.02,
    "mask_image_tail": 0.01,
    "mask_image_head_tail": 0.01,
    "mask_intepolate": 0.01,
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
    type="STDiT3-XL/2",
    from_pretrained=None,
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
