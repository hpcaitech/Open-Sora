# 512x512x64 (65k)
_base_ = [
    "nanosora-f16-256.py",
]

num_frames = 256
frame_interval = 1
data_path = "/home/zhaowangbo/data/HD-VG-130M/processed_data/hdvg_0/out_fmin_256.csv"
image_size = (512, 512)

load = None
ckpt = "PixArt-XL-2-512x512.pth"
additional_model_args = dict(space_scale=0.5, time_scale=1.0)
batch_size = 2

plugin = "zero2-seq"
sequence_parallelism = True
sp_size = 8
