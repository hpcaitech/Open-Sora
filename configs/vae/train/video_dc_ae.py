# ============
# model config 
# ============
model = dict(
    type="dc_ae",
    model_name="dc-ae-f32t4c128",
    from_scratch=True,
    from_pretrained=None,
)

# ============
# data config 
# ============
dataset = dict(
    type="video_text",
    transform_name="resize_crop",
    data_path="datasets/pexels_45k_necessary.csv",
    fps_max=24,
)

bucket_config = {
    "256px_ar1:1": {32: (1.0, 1)},
}

num_bucket_build_workers = 64
num_workers = 12
prefetch_factor = 2

# ============
# train config 
# ============
optim = dict(
    cls="HybridAdam",
    lr=5e-5,
    eps=1e-8,
    weight_decay=0.0,
    adamw_mode=True,
    betas=(0.9, 0.98),
)
lr_scheduler = dict(warmup_steps=0)

mixed_strategy = "mixed_video_image"
mixed_image_ratio = 0.2  # 1:4

dtype = "bf16"
plugin = "zero2"
plugin_config = dict(
    reduce_bucket_size_in_m=128,
    overlap_allgather=False,
)

grad_clip = 1.0
grad_checkpoint = False
pin_memory_cache_pre_alloc_numels = [50 * 1024 * 1024] * num_workers * prefetch_factor

seed = 42
outputs = "outputs"
epochs = 100
log_every = 10
ckpt_every = 3000
keep_n_latest = 50
ema_decay = 0.99
wandb_project = "dcae"

update_warmup_steps = True

# ============
# loss config 
# ============
vae_loss_config = dict(
    perceptual_loss_weight=0.5,
    kl_loss_weight=0,
)

