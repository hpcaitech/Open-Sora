# sample size
num_frames = 16
fps = 24 // 3
image_size = (256, 256)

# model config
model = dict(
    type="STDiT-XL/2",
    space_scale=0.5,
    time_scale=1.0,
    from_pretrained="outputs/129-F16S3-PixArt-ST-XL-2/epoch83-global_step80000/ema.pt",
    # from_pretrained="outputs/285-F16S3-PixArt-ST-XL-2/epoch615-global_step24000/ema.pt",
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="t5",
    from_pretrained="./pretrained_models/t5_ckpts",
    model_max_length=120,
)
scheduler = dict(
    # type="iddpm",
    # num_sampling_steps=250,
    type="dpm-solver",
    num_sampling_steps=20,
    cfg_scale=7.0,
)
dtype = "fp16"

# prompts
batch_size = 2
seed = 42
prompt_path = "./assets/texts/t2v_samples.txt"
save_dir = "./samples/"
