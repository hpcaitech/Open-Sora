_base_ = ["video.py"]

dataset = dict(
    rand_sample_interval=8,
)

bucket_config = {
    "_delete_": True,
    "256px_ar1:1": {33: (1.0, 2)},
}

vae_loss_config = dict(
    perceptual_loss_weight=0.1,
    kl_loss_weight=1e-6,
)

opl_loss_weight = 0

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

discriminator = dict(type="N_Layer_discriminator_3D", from_pretrained=None, input_nc=3, n_layers=5, conv_cls="conv3d")

gen_loss_config = dict(
    gen_start=0,
    disc_factor=1,
    disc_weight=0.05,
)

disc_loss_config = dict(
    disc_start=0,
    disc_factor=1.0,
    disc_loss_type="wgan-gp",
)
