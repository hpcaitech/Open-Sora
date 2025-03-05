_base_ = ["stage1_i2v.py"]

bucket_config = {
    "_delete_": True,
    "256px": {
        129: (1.0, 1),
    },
}
