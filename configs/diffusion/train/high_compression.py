_base_ = ["image.py"]

bucket_config = {
    "_delete_": True,
    "1024px_ar1:1": {16: (1.0, 16)},
}

patch_size = 1
model = dict(
    from_pretrained="/mnt/ddn/sora/tmp_load/vo2_1_768px_t2v_adapt.pt",
    grad_ckpt_settings=None,
    in_channels=512,
)
ae = dict(
    _delete_=True,
    type="dc_ae",
    model_name="dc-ae-f128c512-sana-1.0",
    from_scratch=True,
    from_pretrained="/home/chenli/luchen/Open-Sora-Dev/outputs/250210_125346-vae_train_sana_2d_256channel/epoch0-global_step15500",
)

pin_memory_cache_pre_alloc_numels = [(260 + 20) * 1024 * 1024] * 24 + [
    (34 + 20) * 1024 * 1024
] * 4
lr = 5e-5
optim = dict(
    lr=lr,
)
ema_decay = None
ckpt_every = 500  # save every 4 hours
keep_n_latest = 20
wandb_project = "dcae-adapt"
