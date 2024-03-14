_base_ = [
    "ucf-dit.py",
]

model = "Latte-XL/2"
condition = "label_101"
additional_model_args = dict(condition=condition)
ckpt = "Latte-XL-2-256x256-ucf101.pt"

labels = [0, 1, 2, 3, 4, 5]
