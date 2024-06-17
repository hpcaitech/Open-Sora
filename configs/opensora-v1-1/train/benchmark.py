# this file is only for batch size search and is not used for training

# Define dataset
dataset = dict(
    type="VariableVideoTextDataset",
    data_path=None,
    num_frames=None,
    frame_interval=3,
    image_size=(None, None),
    transform_name="resize_crop",
)

# bucket config format:
# 1. { resolution: {num_frames: (prob, batch_size)} }, in this case batch_size is ignored when searching
# 2. { resolution: {num_frames: (prob, (max_batch_size, ))} }, batch_size is searched in the range [batch_size_start, max_batch_size), batch_size_start is configured via CLI
# 3. { resolution: {num_frames: (prob, (min_batch_size, max_batch_size))} }, batch_size is searched in the range [min_batch_size, max_batch_size)
# 4. { resolution: {num_frames: (prob, (min_batch_size, max_batch_size, step_size))} }, batch_size is searched in the range [min_batch_size, max_batch_size) with step_size (grid search)
# 5. { resolution: {num_frames: (0.0, None)} }, this bucket will not be used

bucket_config = {
    # == manual search ==
    # "240p": {128: (1.0, 2)}, # 4.28s/it
    # "240p": {64: (1.0, 4)},
    # "240p": {32: (1.0, 8)},  # 4.6s/it
    # "240p": {16: (1.0, 16)},  # 4.6s/it
    # "480p": {16: (1.0, 4)},  # 4.6s/it
    # "720p": {16: (1.0, 2)},  # 5.89s/it
    # "256": {1: (1.0, 256)},  # 4.5s/it
    # "512": {1: (1.0, 96)}, # 4.7s/it
    # "512": {1: (1.0, 128)}, # 6.3s/it
    # "480p": {1: (1.0, 50)},  # 4.0s/it
    # "1024": {1: (1.0, 32)},  # 6.8s/it
    # "1024": {1: (1.0, 20)}, # 4.3s/it
    # "1080p": {1: (1.0, 16)}, # 8.6s/it
    # "1080p": {1: (1.0, 8)},  # 4.4s/it
    # == stage 2 ==
    # "240p": {
    #     16: (1.0, (2, 32)),
    #     32: (1.0, (2, 16)),
    #     64: (1.0, (2, 8)),
    #     128: (1.0, (2, 6)),
    # },
    # "256": {1: (1.0, (128, 300))},
    # "512": {1: (0.5, (64, 128))},
    # "480p": {1: (0.4, (32, 128)), 16: (0.4, (2, 32)), 32: (0.0, None)},
    # "720p": {16: (0.1, (2, 16)), 32: (0.0, None)},  # No examples now
    # "1024": {1: (0.3, (8, 64))},
    # "1080p": {1: (0.3, (2, 32))},
    # == stage 3 ==
    "720p": {1: (20, 40), 32: (0.5, (2, 4)), 64: (0.5, (1, 1))},
}


# Define acceleration
num_workers = 4
num_bucket_build_workers = 16
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="STDiT2-XL/2",
    from_pretrained=None,
    input_sq_size=512,  # pretrained model is trained on 512x512
    qk_norm=True,
    qk_norm_legacy=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    micro_batch_size=4,
    local_files_only=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=200,
    shardformer=True,
    local_files_only=True,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# Others
seed = 42
outputs = "outputs"
wandb = False

epochs = 1000
log_every = 10
ckpt_every = 1000
load = None

batch_size = None
lr = 2e-5
grad_clip = 1.0
