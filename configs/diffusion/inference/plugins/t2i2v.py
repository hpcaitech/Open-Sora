use_t2i2v = True

# flux configurations
img_flux = dict(
    type="flux",
    from_pretrained="./ckpts/flux1-dev.safetensors",
    guidance_embed=True,
    # model architecture
    in_channels=64,
    vec_in_dim=768,
    context_in_dim=4096,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    cond_embed=False,  # pass i2v & v2v info, for t2v need this layer too but with x_cond and mask all set to 0
)

img_flux_ae = dict(
    type="autoencoder_2d",
    from_pretrained="./ckpts/flux1-dev-ae.safetensors",
    resolution=256,
    in_channels=3,
    ch=128,
    out_ch=3,
    ch_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    z_channels=16,
    scale_factor=0.3611,
    shift_factor=0.1159,
)
img_resolution = "768px"
