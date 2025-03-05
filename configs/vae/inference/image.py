_base_ = ["video.py"]

save_dir = "samples/vae_img"
dataset = dict(
    type="video_text",
    transform_name="resize_crop",
    data_path="/mnt/ddn/sora/meta/test/image_vae.csv",
)
bucket_config = {
    "512px_ar1:1": {1: (1.0, 1)},
}
