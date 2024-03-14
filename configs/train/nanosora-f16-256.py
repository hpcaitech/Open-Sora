# 256x256x16 (4k)
_base_ = [
    "../datasets/internal.py",
    "../system/model.py",
    "../system/default.py",
]

ckpts = {
    "official": "PixArt-XL-2-512x512.pth",
    "base": "outputs/129-F16S3-PixArt-ST-XL-2/epoch83-global_step80000/ema.pt",
}
ckpt = ckpts["base"]
data_path = "preprocess/pexels_inter4k_fmin_48_rp.csv"

load = None
additional_model_args = dict(space_scale=0.5, time_scale=1.0)
batch_size = 8

model = "PixArt-ST-XL/2"
condition = "t5"
t5_path = "./pretrained_models/t5_ckpts"
vae = "stabilityai/sd-vae-ft-ema"
epochs = 1000
global_seed = 42
outputs = "outputs"
wandb = False
log_every = 10
ckpt_every = 1000
lr = 2e-5
grad_clip = 1.0
