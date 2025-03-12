dtype = "bf16"
batch_size = 1
seed = 42
save_dir = "samples/hunyuanvideo_vae"

plugin = "zero2"
dataset = dict(
    type="video_text",
    transform_name="resize_crop",
    fps_max=16,
    data_path="datasets/pexels_45k_necessary.csv",
)
bucket_config = {
    "512px_ar1:1": {97: (1.0, 1)},
}

num_workers = 24
num_bucket_build_workers = 16
prefetch_factor = 4

model = dict(
    type="hunyuan_vae",
    from_pretrained="./ckpts/hunyuan_vae.safetensors",
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,
    scale_factor=0.476986,
    shift_factor=0,
    use_spatial_tiling=True,
    use_temporal_tiling=True,
    time_compression_ratio=4,
)
