_base_ = ["image.py"]

model = dict(
    _delete_=True,
    type="dc_ae",
    model_name="dc-ae-f32c32-sana-1.0",
    from_scratch=True,
)
vae_loss_config = dict(
    perceptual_loss_weight=0.1,
    kl_loss_weight=0,
)
opl_loss_weight = 0

bucket_config = {
    "_delete_": True,
    "256px_ar1:1": {1: (1.0, 12)},
}
optim = dict(
    lr=1e-4,
)
