num_frames = 1
fps = 1
image_size = (2688, 1408)
# image_size = (2048, 2048)

model = dict(
    type="PixArt-Sigma-XL/2",
    space_scale=4,
    no_temporal_pos_emb=True,
    from_pretrained="PixArt-Sigma-XL-2-2K-MS.pth",
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
    subfolder="vae",
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
)
scheduler = dict(
    type="dpm-solver",
    num_sampling_steps=14,
    cfg_scale=4.5,
)
dtype = "bf16"

# Others
batch_size = 1
seed = 42
prompt_path = "./assets/texts/t2i_sigma.txt"
save_dir = "./samples/samples/"
