_base_ = ["image.py"]

bucket_config = {
    "_delete_": True,
    "768px": {
        1: (1.0, 20),
        16: (1.0, 8),
        20: (1.0, 8),
        24: (1.0, 8),
        28: (1.0, 8),
        32: (1.0, 8),
        36: (1.0, 4),
        40: (1.0, 4),
        44: (1.0, 4),
        48: (1.0, 4),
        52: (1.0, 4),
        56: (1.0, 4),
        60: (1.0, 4),
        64: (1.0, 4),
        68: (1.0, 3),
        72: (1.0, 3),
        76: (1.0, 3),
        80: (1.0, 3),
        84: (1.0, 3),
        88: (1.0, 3),
        92: (1.0, 3),
        96: (1.0, 3),
        100: (1.0, 2),
        104: (1.0, 2),
        108: (1.0, 2),
        112: (1.0, 2),
        116: (1.0, 2),
        120: (1.0, 2),
        124: (1.0, 2),
        128: (1.0, 2),  # 30s
    },
}

condition_config = dict(
    t2v=1,
    i2v_head=7,
)

grad_ckpt_settings = (100, 100)
patch_size = 1
model = dict(
    from_pretrained=None,
    grad_ckpt_settings=grad_ckpt_settings,
    in_channels=128,
    cond_embed=True,
    patch_size=patch_size,
)
ae = dict(
    _delete_=True,
    type="dc_ae",
    model_name="dc-ae-f32t4c128",
    from_pretrained="./ckpts/F32T4C128_AE.safetensors",
    from_scratch=True,
    scaling_factor=0.493,
    use_spatial_tiling=True,
    use_temporal_tiling=True,
    spatial_tile_size=256,
    temporal_tile_size=32,
    tile_overlap_factor=0.25,
)
is_causal_vae = False
ae_spatial_compression = 32

ckpt_every = 250
lr = 3e-5
optim = dict(lr=lr)
