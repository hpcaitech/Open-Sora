# sample size
num_frames = 16
frame_interval = 3
image_size = (256, 256)

# dataset
root = None
data_path = "/mnt/hdd/data/csv/ucf101_videos.csv"
use_image_transform = False
num_workers = 4

# acceleration
dtype = "fp16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# model config
model = dict(
    type="DiT-XL/2",
    from_pretrained="DiT-XL-2-256x256.pt",
    enable_flashattn=True,
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

# runtime
seed = 42
outputs = "outputs"
wandb = False

epochs = 1000
log_every = 10
ckpt_every = 1000
load = None

batch_size = 8
lr = 2e-5
grad_clip = 1.0
