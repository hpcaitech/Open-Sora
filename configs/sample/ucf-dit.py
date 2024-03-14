_base_ = [
    "../datasets/ucf101.py",
]

models = ["DiT-XL/2", "DiT-XL/2x2"]
model = models[0]

vae = "stabilityai/sd-vae-ft-ema"  # 1x8x8
condition = "clip"
additional_model_args = dict(condition=condition)
ckpt = ""

labels = ["Biking", "Cliff Diving", "Rock Climbing Indoor", "Punch", "Tai Chi"]
# labels = ["Apply Eye Makeup", "Apply Lipstick", "Archery", "Baby Crawling", "Balance Beam", "Band Marching"]

global_seed = 42
cfg_scale = 7.0
num_sampling_steps = 250
dtype = "fp32"
