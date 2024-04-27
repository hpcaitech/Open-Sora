num_frames = 1
fps = 1
image_size = (2048, 2048)
multi_resolution = "STDiT2"


# Define model
# model = dict(
#     type="STDiT2-XL/2",
#     from_pretrained="/home/zhouyukun/data/models/PixArt-Sigma/PixArt-Sigma-XL-2-256x256.pth",
#     input_sq_size=512,
#     qk_norm=True,
#     enable_flashattn=True,
#     enable_layernorm_kernel=True,
# )

model = dict(
    type="PixArt-Sigma-XL/2",
    space_scale=4,
    no_temporal_pos_emb=True,
    from_pretrained="PixArt-Sigma-XL-2-2K-MS.pth",
)


vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
    subfolder="vae"
)

text_encoder = dict(
    type="t5",
    from_pretrained="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
    model_max_length=300,
    cache_dir=None,
    subfolder=True
)


scheduler = dict(
    type="iddpm",
    num_sampling_steps=250,
    cfg_scale=7,
    cfg_channel=3,  # or None
)

# scheduler = dict(
#     type="dpm-solver",
#     num_sampling_steps=50,
#     cfg_scale=4.0,
# )

dtype = "bf16"

# Condition
prompt_path = "./assets/texts/t2v_samples.txt"
prompt = None  # prompt has higher priority than prompt_path

# Others
batch_size = 1
seed = 42
save_dir = "./samples/samples/"
