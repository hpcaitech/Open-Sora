_base_ = ["video.py"]

bucket_config = {
    "256px_ar1:1": {1: (1.0, 1)},
}

discriminator = None
gen_loss_confg = None
disc_loss_config = None
disc_lr_scheduler = None
