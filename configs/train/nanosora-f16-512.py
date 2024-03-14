# 512x512x64 (65k)
_base_ = [
    "nanosora-f16-256.py",
]

num_frames = 16
image_size = (512, 512)

load = None
ckpt = "PixArt-XL-2-512x512.pth"
additional_model_args = dict(space_scale=0.5, time_scale=1.0)
batch_size = 2
