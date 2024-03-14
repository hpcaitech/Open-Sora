_base_ = ["IN-dit.py"]

model = "PixArt-XL/2"
condition = "t5"  # [t5, t5_null]
additional_model_args = dict(no_temporal_pos_emb=True)

ckpt = "PixArt-XL-2-256x256.pth"
t5_path = "./pretrained_models/t5_ckpts"

# optimization
lr = 2e-5
batch_size = 64
