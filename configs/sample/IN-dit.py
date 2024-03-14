_base_ = [
    "../datasets/imagenet_256.py",
]

model = "DiT-XL/2"
vae = "stabilityai/sd-vae-ft-ema"  # 1x8x8
condition = "clip"
additional_model_args = dict(condition=condition)

ckpts = {
    "from_scratch": "outputs/004-F1S1-DiT-XL/2/epoch45-global_step57000/ema.pt",
    "align": "outputs/003-F1S1-DiT-XL/2/epoch1-global_step2000",
}
ckpt = ckpts["from_scratch"]

labels = ["golden retriever", "otter", "lesser panda", "geyser", "macaw", "valley", "balloon", "golden panda"]
global_seed = 42
cfg_scale = 4.0
num_sampling_steps = 250
dtype = "fp32"
