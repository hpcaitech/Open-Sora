_base_ = [
    "../datasets/internal.py",
    "../system/model.py",
    "../system/default.py",
]

model = "PixArt-XL/2"
condition = "t5"  # [t5, t5_null]
t5_path = "./pretrained_models/t5_ckpts"
vae = "stabilityai/sd-vae-ft-ema"

ckpt = "PixArt-XL-2-512x512.pth"
additional_model_args = dict(space_scale=0.5, time_scale=1.0)

load = None
epochs = 1000
global_seed = 42
outputs = "outputs"
wandb = False
log_every = 10
ckpt_every = 1000

# optimization
lr = 2e-5
grad_clip = 1.0
batch_size = 8
