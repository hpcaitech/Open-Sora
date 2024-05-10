# Define dataset
dataset = dict(
    type="VideoTextDataset",
    data_path=None,
    num_frames=1,
    frame_interval=1,
    image_size=(256, 256),
    transform_name="center",
)

# Define acceleration
num_workers = 4
dtype = "bf16"
grad_checkpoint = False
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="DiT-XL/2",
    no_temporal_pos_emb=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="clip",
    from_pretrained="openai/clip-vit-base-patch32",
    model_max_length=77,
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
ckpt_every = 1000
load = None

batch_size = 128
lr = 1e-4  # according to DiT repo
grad_clip = 1.0
