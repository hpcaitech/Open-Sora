# 256x256x64 (16k)
_base_ = [
    "nanosora-f16-256.py",
]

num_frames = 64
frame_interval = 2
data_path = "preprocess/out_length_130_250_withoutbroken.csv"
grad_checkpoint = True

load = None
ckpt = "outputs/129-F16S3-PixArt-ST-XL-2/epoch83-global_step80000/ema.pt"
additional_model_args = dict(space_scale=0.5, time_scale=2 / 3)
batch_size = 2
