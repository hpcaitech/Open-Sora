_base_ = [
    "ucf-dit.py",
]

models = ["DiT-STP-XL/2", "DiT-ST-XL/2"]
model = models[0]

ckpts = {
    "old_st": "../minisora/outputs/326-F16S3-STDiT-XL-1x2x2/epoch384-global_step80000/ema.pt",
    "stv": "outputs/020-F16S3-DiT-STV-XL-2/epoch173-global_step36000/",
    "st_freeze": "outputs/025-F16S3-DiT-ST-XL-2/epoch206-global_step43000/",
    "st_freeze_cont": "outputs/036-F16S3-DiT-ST-XL-2/epoch76-global_step16000/",
}
ckpt = ckpts["st_freeze_cont"]
