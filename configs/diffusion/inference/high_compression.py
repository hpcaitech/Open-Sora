_base_ = ["t2i2v_768px.py"]

# no need for parallelism
plugin = None
plugin_config = None
plugin_ae = None
plugin_config_ae = None

# model settings
patch_size = 1
model = dict(
    in_channels=128,
    cond_embed=True,
    patch_size=1,
)

# AE settings
ae = dict(
    _delete_=True,
    type="dc_ae",
    from_scratch=True,
    model_name="dc-ae-f32t4c128",
    from_pretrained="/mnt/jfs-hdd/sora/checkpoints/shenchenhui/video_sana_128c/250221_102256-vae_train_video_dc_ae_tempcompress_disc/epoch0-global_step459000",
    use_spatial_tiling=True,
    use_temporal_tiling=True,
    spatial_tile_size=256,
    temporal_tile_size=32,
    tile_overlap_factor=0.25,
)
ae_spatial_compression = 32
