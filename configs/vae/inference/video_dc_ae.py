dtype = "bf16"
batch_size = 1
seed = 42

dataset = dict(
    type="video_text",
    transform_name="resize_crop",
    fps_max=16,
    data_path="datasets/pexels_45k_necessary.csv",
)
bucket_config = {
    "512px_ar1:1": {96: (1.0, 1)},
}

model = dict(
    type="dc_ae",
    model_name="dc-ae-f32t4c128",
    from_pretrained="./ckpts/F32T4C128_AE.safetensors",
    from_scratch=True,
    use_spatial_tiling=True,
    use_temporal_tiling=True,
    spatial_tile_size=256,
    temporal_tile_size=32,
    tile_overlap_factor=0.25,
)

save_dir = "samples/video_dc_ae"

num_workers = 24
num_bucket_build_workers = 16
prefetch_factor = 4

