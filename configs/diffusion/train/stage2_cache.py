_base_ = ["stage2.py"]

# Dataset settings
cached_text = False
cached_video = True
dataset = dict(
    type="cached_video_text",
    transform_name="resize_crop",
    fps_max=24,
    cached_text=cached_text,
    cached_video=cached_video,
)


# == stage2: 256px ==
bucket_config = {
    "768px": {
        1: (1.0, 38),
        5: (1.0, 6),
        9: (1.0, 6),
        13: (1.0, 6),
        17: (1.0, 6),
        21: (1.0, 6),
        25: (1.0, 6),
        29: (1.0, 6),
        33: (1.0, 6),
        37: (1.0, 4),
        41: (1.0, 4),
        45: (1.0, 4),
        49: (1.0, 4),
        53: (1.0, 4),
        57: (1.0, 4),
        61: (1.0, 4),
        65: (1.0, 4),
        69: (1.0, 3),
        73: (1.0, 3),
        77: (1.0, 3),
        81: (1.0, 3),
        85: (1.0, 3),
        89: (1.0, 3),
        93: (1.0, 3),
        97: (1.0, 3),
        101: (1.0, 2),
        105: (1.0, 2),
        109: (1.0, 2),
        113: (1.0, 2),
        117: (1.0, 2),
        121: (1.0, 2),
        125: (1.0, 2),
        129: (1.0, 2),
    },
}
pin_memory_cache_pre_alloc_numels = [(260 + 20) * 1024 * 1024] * 24 + [(34 + 20) * 1024 * 1024] * 4
# record_time = True
# record_barrier = True
