num_frames = 113
fps = 24
frame_interval = 1
resolution = "720p"
aspect_ratio = "9:16"

save_dir = "./samples/samples/"
multi_resolution = "STDiT2"
seed = 42
batch_size = 1
dtype = "bf16"

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="/home/guoxinying/open_source_video_ocean_V1/OpenSora-STDiT-v4",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    kernel_size=(8, 8, -1),
    use_spatial_rope=True,
    force_huggingface=True,
)
vae = dict(
    type="OpenSoraVAE_V1_3",
    from_pretrained="/home/guoxinying/open_source_video_ocean_V1/OpenSora-VAE-v1.3",
    z_channels=16,
    micro_batch_size=1,
    micro_batch_size_2d=4,
    micro_frame_size=17,
    use_tiled_conv3d=True,
    tile_size=4,
    normalization="video",
    temporal_overlap=True,
    force_huggingface=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="/mnt/jfs-hdd/sora/checkpoints/pretrained_models/t5-v1_1-xxl",
    model_max_length=300,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.5,
    use_oscillation_guidance=True,
    use_flaw_fix=True,
    # transform_scale=2.0,
)
