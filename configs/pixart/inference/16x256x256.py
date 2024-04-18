num_frames = 16
fps = 8
image_size = (256, 256)

# Define model
model = dict(
    type="PixArt-XL/2",
    space_scale=0.5,
    time_scale=1.0,
    from_pretrained="outputs/098-F16S3-PixArt-XL-2/epoch7-global_step30000/model_ckpt.pt",
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=120,
)
scheduler = dict(
    type="dpm-solver",
    num_sampling_steps=20,
    cfg_scale=7.0,
)
dtype = "bf16"

# Others
batch_size = 2
seed = 42
prompt_path = "./assets/texts/t2v_samples.txt"
save_dir = "./samples/samples/"
