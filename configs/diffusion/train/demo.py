_base_ = ["stage1.py"]


bucket_config = {
    "_delete_": True,
    "256px": {
        1: (1.0, 1),
        33: (1.0, 1),
        97: (1.0, 1),
        129: (1.0, 1),
    },
}
