_base_ = [
    "ucf-dit.py",
]

models = ["DiT-STP-XL/2", "DiT-ST-XL/2"]
model = models[0]
# additional_model_args["freeze"] = "non_temporal"

ckpts = {
    "official": "DiT-XL-2-256x256.pt",
    "st_freeze": "outputs/025-F16S3-DiT-ST-XL-2/epoch206-global_step43000/model_ckpt.pt",
}
ckpt = ckpts["st_freeze"]
