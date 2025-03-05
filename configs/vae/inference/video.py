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
    type="autoencoder_3d",
    from_pretrained=None,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    z_channels=16,
    scale_factor=1.0,
    shift_factor=0.0,
    tiling=4,
)

num_workers = 24
num_bucket_build_workers = 16
prefetch_factor = 4
