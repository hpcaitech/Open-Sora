_base_ = ["hunyuan_video_disc.py"]

model = dict(
    encoder_add_residual=True,
    encoder_slice_t=True,
    decoder_add_residual=True,
    decoder_slice_t=True,
)

mixed_image_ratio = 0.25
