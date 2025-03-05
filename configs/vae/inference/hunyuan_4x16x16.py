_base_ = ["hunyuan_video.py"]


model = dict(
    _delete_=True,
    type="hunyuan_vae",
    from_pretrained=None,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,
    dropout=0.0,
    block_out_channels=(128, 256, 256, 512, 512),
    time_compression_ratio=4,
    spatial_compression_ratio=16,
    encoder_add_residual=True,
    encoder_slice_t=True,
    decoder_add_residual=True,
    decoder_slice_t=True,
)
