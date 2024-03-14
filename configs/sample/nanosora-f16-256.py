_base_ = [
    "../datasets/ucf101.py",
    "prompts.py",
]

model = "PixArt-ST-XL/2"
condition = "t5"
additional_model_args = dict(space_scale=0.5)
# vae = "stabilityai/sd-vae-ft-ema"
vae = "vae_temporal_decoder"

ckpts = {
    "early_2e-5": "outputs/097-F16S3-PixArt-ST-XL-2/epoch0-global_step1000/",
    "latest_2e-5": "outputs/097-F16S3-PixArt-ST-XL-2/epoch7-global_step30000/",
    "latest_1e-4": "outputs/099-F16S3-PixArt-ST-XL-2/epoch7-global_step30000/",
    "nanosora_30k": "outputs/129-F16S3-PixArt-ST-XL-2/epoch31-global_step30000/ema.pt",
    "nanosora_63k": "outputs/129-F16S3-PixArt-ST-XL-2/epoch66-global_step63000/ema.pt",
    "nanosora_70k": "outputs/129-F16S3-PixArt-ST-XL-2/epoch73-global_step70000/ema.pt",
    "nanosora_80k": "outputs/129-F16S3-PixArt-ST-XL-2/epoch83-global_step80000/ema.pt",
    "nanosora_90k": "outputs/129-F16S3-PixArt-ST-XL-2/epoch94-global_step90000/ema.pt",
    "nanosora_100k": "outputs/129-F16S3-PixArt-ST-XL-2/epoch104-global_step100000/ema.pt",
    "nanosora_hq_2k": "outputs/285-F16S3-PixArt-ST-XL-2/epoch51-global_step2000/",
    "nanosora_hq_4k": "outputs/285-F16S3-PixArt-ST-XL-2/epoch102-global_step4000/ema.pt",
    "nanosora_hq_8k": "outputs/285-F16S3-PixArt-ST-XL-2/epoch205-global_step8000/ema.pt",
    "nanosora_hq_15k": "outputs/285-F16S3-PixArt-ST-XL-2/epoch384-global_step15000/",
    "nanosora_hq_20k": "outputs/285-F16S3-PixArt-ST-XL-2/epoch512-global_step20000/",
}
ckpt = ckpts["nanosora_hq_20k"]
t5_path = "./pretrained_models/t5_ckpts"

global_seed = 42
cfg_scale = 7.0
num_sampling_steps = 250
dtype = "fp16"
fps = 8
