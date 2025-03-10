_base_ = ["t2i2v_768px.py"]

# no need for parallelism
plugin = None
plugin_config = None
plugin_ae = None
plugin_config_ae = None

# model settings
patch_size = 1
model = dict(
    from_pretrained="./ckpts/Open_Sora_v2_Video_DC_AE.safetensors",
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
    from_pretrained="./ckpts/F32T4C128_AE.safetensors",
    use_spatial_tiling=True,
    use_temporal_tiling=True,
    spatial_tile_size=256,
    temporal_tile_size=32,
    tile_overlap_factor=0.25,
)
ae_spatial_compression = 32

sampling_option = dict(
    num_frames=128,
)
