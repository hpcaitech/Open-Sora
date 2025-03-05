_base_ = ["image.py"]

dataset = dict(
    memory_efficient=True,
)

# new config
grad_ckpt_settings = (8, 100)
bucket_config = {
    "_delete_": True,
    "256px": {
        1: (1.0, 45),  # 6.22 s
        5: (1.0, 12),
        9: (1.0, 12),
        13: (1.0, 12),
        17: (1.0, 12),
        21: (1.0, 12),
        25: (1.0, 12),
        29: (1.0, 12),
        33: (1.0, 12),  # 7.02 s
        37: (1.0, 6),
        41: (1.0, 6),
        45: (1.0, 6),
        49: (1.0, 6),
        53: (1.0, 6),
        57: (1.0, 6),
        61: (1.0, 6),
        65: (1.0, 6),  # 6.79 s
        69: (1.0, 4),
        73: (1.0, 4),
        77: (1.0, 4),
        81: (1.0, 4),
        85: (1.0, 4),
        89: (1.0, 4),
        93: (1.0, 4),
        97: (1.0, 4),  # 6.84 s
        101: (1.0, 3),
        105: (1.0, 3),
        109: (1.0, 3),
        113: (1.0, 3),
        117: (1.0, 3),
        121: (1.0, 3),
        125: (1.0, 3),
        129: (1.0, 3),  # 7.48 s
    },
    "768px": {
        1: (0.5, 13),
    },
    "1024px": {
        1: (0.5, 7),
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
ema_decay = 0.999
ckpt_every = 2000  # save every 4 hours
keep_n_latest = 20
wandb_project = "mmdit-vo3"
async_io = False
