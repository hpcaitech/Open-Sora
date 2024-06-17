# Define dataset
# dataset = dict(
#     type="VariableVideoTextDataset",
#     data_path=None,
#     num_frames=None,
#     frame_interval=3,
#     image_size=(None, None),
#     transform_name="resize_crop",
# )
dataset = dict(
    type="VideoTextDataset",
    data_path=None,
    num_frames=1,
    frame_interval=1,
    image_size=(256, 256),
    transform_name="center",
)
bucket_config = {  # 6s/it
    "256": {1: (1.0, 256)},
    "512": {1: (1.0, 80)},
    "480p": {1: (1.0, 52)},
    "1024": {1: (1.0, 20)},
    "1080p": {1: (1.0, 8)},
}

# Define acceleration
num_workers = 16
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
# model = dict(
#     type="DiT-XL/2",
#     from_pretrained="/home/zhaowangbo/wangbo/PixArt-alpha/pretrained_models/PixArt-XL-2-512x512.pth",
#     # input_sq_size=512,  # pretrained model is trained on 512x512
#     enable_flash_attn=True,
#     enable_layernorm_kernel=True,
# )
model = dict(
    type="PixArt-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    no_temporal_pos_emb=True,
    from_pretrained="PixArt-XL-2-512x512.pth",
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
# model = dict(
#     type="DiT-XL/2",
#     # space_scale=1.0,
#     # time_scale=1.0,
#     no_temporal_pos_emb=True,
#     # from_pretrained="PixArt-XL-2-512x512.pth",
#     from_pretrained="/home/zhaowangbo/wangbo/PixArt-alpha/pretrained_models/PixArt-XL-2-512x512.pth",
#     enable_flash_attn=True,
#     enable_layernorm_kernel=True,
# )
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    micro_batch_size=4,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=200,
    shardformer=True,
)
scheduler = dict(
    type="rflow",
    # timestep_respacing="",
)

# Others
seed = 42
outputs = "outputs"
wandb = False

epochs = 10
log_every = 10
ckpt_every = 500
load = None

batch_size = 100  # only for logging
lr = 2e-5
grad_clip = 1.0
