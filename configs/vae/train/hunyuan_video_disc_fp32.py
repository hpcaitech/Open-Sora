_base_ = ["hunyuan_video_disc.py"]

bucket_config = {
    "_delete_": True,
    "256px_ar1:1": {33: (1.0, 1)},
    # "256px_ar1:1": {65: (1.0, 1)},
}

ema_decay = None
