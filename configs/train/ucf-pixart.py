_base_ = [
    "ucf-dit.py",
]

model = "PixArt-XL/2"
condition = "t5"  # [t5, t5_null]
additional_model_args = dict(_delete_=True)
ckpt = "PixArt-XL-2-256x256.pth"
t5_path = "./pretrained_models/t5_ckpts"
