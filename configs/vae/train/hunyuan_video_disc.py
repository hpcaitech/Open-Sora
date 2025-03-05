_base_ = ["hunyuan_video.py"]

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
