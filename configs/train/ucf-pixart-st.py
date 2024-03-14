_base_ = [
    "ucf-pixart.py",
]

model = "PixArt-ST-XL/2"
condition = "t5"  # [t5, t5_null]
t5_path = "./pretrained_models/t5_ckpts"
ckpt = "PixArt-XL-2-512x512.pth"
additional_model_args = dict(space_scale=0.5, time_scale=1.0)

load = None
batch_size = 6
