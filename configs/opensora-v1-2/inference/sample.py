image_size = (240, 426)
num_frames = 51
fps = 24
frame_interval = 1

prompt_path = "./assets/texts/t2v_samples.txt"
save_dir = "./samples/samples/"
seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"

model = dict(
    type="STDiT3-XL/2",
    from_pretrained=None,
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderPipeline",
    from_pretrained="pretrained_models/vae-v3",
    # scale=2.5,
    shift=(-0.10, 0.34, 0.27, 0.98),
    scale=(3.85, 2.32, 2.33, 3.06),
    micro_frame_size=17,
    vae_2d=dict(
        type="VideoAutoencoderKL",
        from_pretrained="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
        subfolder="vae",
        micro_batch_size=4,
        local_files_only=True,
    ),
    vae_temporal=dict(
        type="VAE_Temporal_SD",
        from_pretrained=None,
    ),
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
    num_sampling_steps=30,
    cfg_scale=7.0,
)
