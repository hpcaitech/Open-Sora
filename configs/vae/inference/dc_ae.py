dtype = "bf16"
batch_size = 1
seed = 42
save_dir = "samples/vae_vid"

plugin = "zero2"
dataset = dict(
    type="video_text",
    transform_name="resize_crop",
    fps_max=16,
    data_path="/mnt/ddn/sora/meta/test/vid_vae.csv",
)
bucket_config = {
    "256px_ar1:1": {32: (1.0, 1)},
}

model = dict(
    type="dc_ae",
    # model_name="mit-han-lab/dc-ae-f32c32-sana-1.0",
    # model_name="dc-ae-f32c32-sana-1.0",
    model_name="dc-ae-f128c512-sana-1.0",
    from_scratch=True,
)

num_workers = 24
num_bucket_build_workers = 16
prefetch_factor = 4
