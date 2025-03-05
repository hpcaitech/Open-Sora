_base_ = ["256px.py"]

vae_loss_config = dict(
    perceptual_loss_weight=0.1,
)
gen_loss_config = dict(
    disc_factor=0.0,
)
disc_loss_config = dict(
    disc_factor=0.0,
)
discriminator = None
