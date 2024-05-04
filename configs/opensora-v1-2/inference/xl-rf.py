num_frames = 1
fps = 1
image_size = (2560, 1536)
# image_size = (2048, 2048)
multi_resolution = "STDiT2"

model = dict(
    type="STDiT3-XL/2",
    from_pretrained=None,
    qk_norm=True,
    enable_flashattn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    micro_batch_size=4,
    from_pretrained="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
    subfolder="vae",
    local_files_only=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
    local_files_only=True,
)
scheduler = dict(
    type="rflow",
    use_discrete_timesteps=False,
    use_timestep_transform=True,
    num_sampling_steps=8,
    cfg_scale=4.5,
)
dtype = "bf16"

# Others
batch_size = 1
seed = 42
prompt_path = "./assets/texts/t2i_sigma.txt"
save_dir = "./samples/samples/"
