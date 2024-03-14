_base_ = [
    "nanosora-f16-256.py",
]

additional_model_args = dict(space_scale=0.5, time_scale=2 / 3)
ckpt = "outputs/172-F64S2-PixArt-ST-XL-2/epoch16-global_step20000"

num_frames = 64
frame_interval = 2
fps = 12
