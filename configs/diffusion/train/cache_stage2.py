to_cache_text = False
to_cache_video = True
cached_text = False
cached_video = False
# exist_handling = "ignore"

dataset = dict(
    type="cached_video_text",
    transform_name="resize_crop",
    fps_max=24,
    return_latents_path=to_cache_text or to_cache_video,
    cached_text=False,
    cached_video=False,
)

bucket_config = {
    "768px": {
        1: (1.0, 50),  # 20s/it
        5: (1.0, 15),
        9: (1.0, 15),
        13: (1.0, 15),
        17: (1.0, 15),
        21: (1.0, 15),
        25: (1.0, 15),
        29: (1.0, 15),
        33: (1.0, 15),
        37: (1.0, 10),
        41: (1.0, 10),
        45: (1.0, 10),
        49: (1.0, 10),
        53: (1.0, 10),
        57: (1.0, 10),
        61: (1.0, 10),
        65: (1.0, 10),
        69: (1.0, 7),
        73: (1.0, 7),
        77: (1.0, 7),
        81: (1.0, 7),
        85: (1.0, 7),
        89: (1.0, 7),
        93: (1.0, 7),
        97: (1.0, 7),
        101: (1.0, 6),
        105: (1.0, 6),
        109: (1.0, 6),
        113: (1.0, 6),
        117: (1.0, 6),
        121: (1.0, 6),
        125: (1.0, 6),
        129: (1.0, 6),  # 46s
    },
}
pin_memory_cache_pre_alloc_numels = [(260 + 20) * 1024 * 1024] * 24 + [(34 + 20) * 1024 * 1024] * 4
# record_time = True
# record_barrier = True

ae = dict(
    type="hunyuan_vae",
    from_pretrained="/mnt/jfs-hdd/sora/checkpoints/pretrained_models/hunyuan-video-t2v-720p/vae/pytorch_model.pt",
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,
    use_spatial_tiling=True,
    use_temporal_tiling=False,
)
t5 = dict(
    type="text_embedder",
    from_pretrained="google/t5-v1_1-xxl",
    cache_dir="/mnt/ddn/sora/tmp_load/huggingface/hub/",
    max_length=512,
    shardformer=True,
)
clip = dict(
    type="text_embedder",
    from_pretrained="openai/clip-vit-large-patch14",
    cache_dir="/mnt/ddn/sora/tmp_load/huggingface/hub/",
    max_length=77,
)

# Acceleration settings
prefetch_factor = 2
num_workers = 6
num_bucket_build_workers = 64
dtype = "bf16"

# Other settings
seed = 42
wandb_project = "mmdit"
