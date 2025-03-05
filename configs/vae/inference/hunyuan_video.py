_base_ = ["video.py"]

model = dict(
    _delete_=True,
    type="hunyuan_vae",
    from_pretrained="/mnt/jfs-hdd/sora/checkpoints/pretrained_models/hunyuan-video-t2v-720p/vae/pytorch_model.pt",
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,
    scale_factor=0.476986,
    shift_factor=0,
    use_spatial_tiling=True,
    use_temporal_tiling=True,
    # set the following to True to use residual
    encoder_add_residual=False,
    decoder_add_residual=False,
    # residual slice or pad
    encoder_slice_t=False,
    decoder_slice_t=False,
    # temporal
    time_compression_ratio=4,
)
