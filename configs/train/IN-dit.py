_base_ = [
    "../datasets/imagenet_256.py",  # dataset config
    "../system/model.py",  # acceleration for model
    "../system/default.py",  # acceleration generally
]

# one image equals 1x256x256 / 1x8x8 / 1x2x2 = 256 tokens

model = "DiT-XL/2"
vae = "stabilityai/sd-vae-ft-ema"  # 1x8x8, [stabilityai/sd-vae-ft-ema, stabilityai/sd-vae-ft]
condition = "clip"
additional_model_args = dict(condition=condition)

load = None
epochs = 1000
global_seed = 42
outputs = "outputs"
wandb = False
log_every = 10
ckpt_every = 1000

# optimization
lr = 1e-4
grad_clip = 1.0
batch_size = 128
