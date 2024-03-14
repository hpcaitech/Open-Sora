_base_ = [
    "../datasets/ucf101.py",
    "../system/model.py",
    "../system/default.py",
]

models = ["DiT-XL/2", "DiT-XL/2x2"]
model = models[0]

vae = "stabilityai/sd-vae-ft-ema"
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
lr = 2e-5
grad_clip = 1.0
batch_size = 8
