_base_ = ["image.py"]

# new config
grad_ckpt_settings = (100, 100)  # one GPU
# grad_ckpt_buffer_size = 20 * 1024**3
plugin = "hybrid"
plugin_config = dict(
    tp_size=1,
    pp_size=1,
    sp_size=4,
    sequence_parallelism_mode="ring_attn",
    enable_sequence_parallelism=True,
    static_graph=True,
    zero_stage=2,
)

bucket_config = {
    "_delete_": True,
    "256px": {
        1: (1.0, 130),
        5: (1.0, 14),
        9: (1.0, 14),
        13: (1.0, 14),
        17: (1.0, 14),
        21: (1.0, 14),
        25: (1.0, 14),
        29: (1.0, 14),
        33: (
            1.0,
            14,
        ),  # 7.02 s iter: 4.17 s | encode_video: 1.42 s | encode_text: 0.29 s | forward: 0.53 s | backward: 1.67 s | 135GB
        37: (1.0, 10),
        41: (1.0, 10),
        45: (1.0, 10),
        49: (1.0, 10),
        53: (1.0, 10),
        57: (1.0, 10),
        61: (1.0, 10),
        65: (
            1.0,
            10,
        ),  # 6.79 s iter: 10.42 s | encode_video: 4.02 s | encode_text: 0.43 s | forward: 1.31 s | backward: 4.21 s ｜ 125GB
        69: (1.0, 7),
        73: (1.0, 7),
        77: (1.0, 7),
        81: (1.0, 7),
        85: (1.0, 7),
        89: (1.0, 7),
        93: (1.0, 7),
        97: (
            1.0,
            7,
        ),  # 6.84 s iter: 5.26 s | encode_video: 2.16 s | encode_text: 0.16 s | forward: 0.64 s | backward: 2.08 s | 127GB
        101: (1.0, 6),
        105: (1.0, 6),
        109: (1.0, 6),
        113: (1.0, 6),
        117: (1.0, 6),
        121: (1.0, 6),
        125: (1.0, 6),
        129: (
            1.0,
            6,
        ),  # 7.48 s iter: 9.67 s | encode_video: 3.78 s | encode_text: 0.21 s | forward: 1.36 s | backward: 2.78 s | 130.3GB
    },
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

model = dict(
    from_pretrained=None,
    grad_ckpt_settings=grad_ckpt_settings,
)
lr = 5e-5
optim = dict(
    lr=lr,
)
ema_decay = 0.99
ckpt_every = 200
keep_n_latest = 20
wandb_project = "mmdit-vo3"
