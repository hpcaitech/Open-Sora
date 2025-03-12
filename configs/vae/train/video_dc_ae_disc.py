_base_ = ["video_dc_ae.py"]

discriminator = dict(
    type="N_Layer_discriminator_3D",
    from_pretrained=None,
    input_nc=3,
    n_layers=5,
    conv_cls="conv3d"
)
disc_lr_scheduler = dict(warmup_steps=0)

gen_loss_config = dict(
    gen_start=0,
    disc_weight=0.05,
)

disc_loss_config = dict(
    disc_start=0,
    disc_loss_type="hinge",
)

optim_discriminator = dict(
    cls="HybridAdam",
    lr=1e-4,
    eps=1e-8,
    weight_decay=0.0,
    adamw_mode=True,
    betas=(0.9, 0.98),
)

grad_checkpoint = True
model = dict(
    disc_off_grad_ckpt = True, # set to true if your `grad_checkpoint` is True
)
