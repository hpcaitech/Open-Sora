_base_ = [
    "IN-dit.py",
]

condition = "label_1000"
additional_model_args = dict(condition=condition, no_temporal_pos_emb=True)
ckpt = "DiT-XL-2-256x256.pt"  # DiT official checkpoint

labels = [207, 360, 387, 974, 88, 979, 417, 279]
