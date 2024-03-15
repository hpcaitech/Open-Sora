# sample size
num_frames = 16
fps = 24 // 3
image_size = (512, 512)

# model config
model = dict(
    type="STDiT-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    from_pretrained="outputs/314-F16S3-PixArt-ST-XL-2/epoch128-global_step20000/ema.pt",
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    split=8,
)
text_encoder = dict(
    type="t5",
    from_pretrained="./pretrained_models/t5_ckpts",
    model_max_length=120,
)
scheduler = dict(
    type="iddpm",
    num_sampling_steps=100,
    # type = "dpm-solver",
    # num_sampling_steps=20,
    cfg_scale=7.0,
)
dtype = "fp16"

# prompts
batch_size = 2
seed = 42
prompt_path = "./assets/texts/t2v_samples.txt"
save_dir = "./samples/"
