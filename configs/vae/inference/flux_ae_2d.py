dtype = "bf16"
batch_size = 1
seed = 42
save_dir = "samples/flux_ae_2d"

dataset = dict(
    type="video_text",
    transform_name="resize_crop",
    fps_max=16,
    data_path="/mnt/jfs-hdd/sora/meta/validation/img_1k.csv",
)

bucket_config = {
    "1024px_ar1:1": {1: (1.0, 1)},
}

model = dict(
    type="autoencoder_2d",
    from_pretrained="pretrained_models/flux1-dev/ae.safetensors",
    resolution=256,
    in_channels=3,
    ch=128,
    out_ch=3,
    ch_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    z_channels=16,
    scale_factor=1.0,
    shift_factor=0.0,
)
