# sample size
num_frames = 16
frame_interval = 3
image_size = (512, 512)

# dataset
root = None
data_path = "/home/zhaowangbo/data_hdd/csv/inter4k_pexels_rp_fmin_48.csv"
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
    time_scale=1.0,
    from_pretrained="outputs/285-F16S3-PixArt-ST-XL-2/epoch615-global_step24000/ema.pt",
    enable_flashattn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    split=4,
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
ckpt_every = 500
load = None

batch_size = 8
lr = 2e-5
grad_clip = 1.0
