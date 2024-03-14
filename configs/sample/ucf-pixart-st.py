_base_ = [
    "ucf-dit.py",
]

model = "PixArt-ST-XL/2"
condition = "t5"
additional_model_args = dict(_delete_=True)

ckpts = {
    "official": "PixArt-ST-XL-2-256x256.pth",
    "ucf": "outputs/054-F16S3-PixArt-ST-XL-2/epoch21-global_step6000/",
}
ckpt = ckpts["ucf"]
t5_path = "./pretrained_models/t5_ckpts"
