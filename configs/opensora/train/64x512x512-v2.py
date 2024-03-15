# sample size
num_frames = 64
frame_interval = 2
image_size = (512, 512)

# dataset
root = None
data_path = "/mnt/hdd/data/csv/inter4k_pexels_rp_fmin_128.csv"
use_image_transform = False
num_workers = 4

# acceleration
dtype = "fp16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# model config
model = dict(
    type="STDiT-XL/2",
    space_scale=1.0,
    time_scale=2 / 3,
    from_pretrained="outputs/314-F16S3-PixArt-ST-XL-2/epoch128-global_step20000/ema.pt",
    enable_flashattn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    split=8,
)
text_encoder = dict(
    type="t5",
    from_pretrained="./pretrained_models/t5_ckpts",
    model_max_length=120,
    shardformer=True,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# runtime
seed = 42
outputs = "outputs"
wandb = False

epochs = 1000
log_every = 10
ckpt_every = 250
load = None

batch_size = 4
lr = 2e-5
grad_clip = 1.0
