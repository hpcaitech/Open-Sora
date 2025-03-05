_base_ = ["video.py"]

dataset = dict(
    rand_sample_interval=8,
)

bucket_config = {
    "_delete_": True,
    "256px_ar1:1": {33: (1.0, 2)},
    # "256px_ar1:1": {65: (1.0, 1)},
}

vae_loss_config = dict(
    perceptual_loss_weight=0.1,
    kl_loss_weight=1e-6,
)

opl_loss_weight = 0

model = dict(
    _delete_=True,
    type="hunyuan_vae",
    # from_pretrained="/mnt/jfs-hdd/sora/checkpoints/pretrained_models/hunyuan-video-t2v-720p/vae/pytorch_model.pt",
    from_pretrained=None,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,
    dropout=0.0,  # TODO: expr with 0.1
)
