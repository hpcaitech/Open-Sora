# Define dataset
dataset = dict(
    type="VideoTextDataset",
    data_path=None,
    num_frames=1,
    frame_interval=3,
    image_size=(512, 512),
)

# Define acceleration
num_workers = 4
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="PixArt-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    no_temporal_pos_emb=True,
    # from_pretrained="PixArt-XL-2-512x512.pth",
    from_pretrained="PRETRAINED_MODEL",
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=120,
    shardformer=True,
)
scheduler = dict(
    type="rflow",
    # timestep_respacing="",
)

# Others
seed = 42
outputs = "outputs"
wandb = True

epochs = 2
log_every = 10
ckpt_every = 1000
load = None

batch_size = 64
lr = 2e-5
grad_clip = 1.0
