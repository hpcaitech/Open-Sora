num_frames = 1
fps = 1
image_size = (256, 256)

# Define model
model = dict(
    type="DiT-XL/2",
    no_temporal_pos_emb=True,
    condition="text",
    from_pretrained="PRETRAINED_MODEL",
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="clip",
    from_pretrained="openai/clip-vit-base-patch32",
    model_max_length=77,
)
scheduler = dict(
    type="dpm-solver",
    num_sampling_steps=20,
    cfg_scale=4.0,
)
dtype = "bf16"

# Others
batch_size = 2
seed = 42
prompt_path = "./assets/texts/imagenet_labels.txt"
save_dir = "./samples/samples/"
