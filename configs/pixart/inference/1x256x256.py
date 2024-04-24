num_frames = 1
fps = 1
image_size = (256, 256)

# Define model
model = dict(
    type="PixArt-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    no_temporal_pos_emb=True,
    from_pretrained="PixArt-XL-2-256x256.pth",
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
prompt_path = "./assets/texts/t2i_samples.txt"
save_dir = "./samples/samples/"
