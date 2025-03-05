_base_ = ["video.py"]

model = dict(
    _delete_=True,
    type="hunyuan_vae",
    from_pretrained=None,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=32,
    scale_factor=0.476986,
    shift_factor=0,
    use_spatial_tiling=True,
    use_temporal_tiling=True,
    # architecture
    channel=True,
    time_compression_ratio=4,
    spatial_compression_ratio=32,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    # set the following to True to use residual
    encoder_add_residual=True,
    decoder_add_residual=True,
    # residual slice or pad
    encoder_slice_t=False,
    decoder_slice_t=True,
)
