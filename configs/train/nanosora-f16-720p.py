# 720px64 (230k)
_base_ = [
    "nanosora-f16-256.py",
]

num_frames = 16
# image_size = (720, 1280)
image_size = (960, 960)

load = None
ckpt = "PixArt-XL-2-512x512.pth"
additional_model_args = dict(space_scale=0.5, time_scale=1.0)
batch_size = 1

plugin = "zero2-seq"
sequence_parallelism = True
sp_size = 2
