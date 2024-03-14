# Align official checkpoints with clip text encoder

_base_ = [
    "IN-dit.py",
]

additional_model_args["no_temporal_pos_emb"] = True
ckpt = "DiT-XL-2-256x256.pt"  # load from official checkpoint
