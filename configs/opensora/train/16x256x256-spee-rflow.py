# Define dataset
dataset = dict(
    type="VideoTextDataset",
    data_path=None,
    num_frames=16,
    frame_interval=3,
    image_size=(256, 256),
)

# Define acceleration
num_workers = 4
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=0.5,
    time_scale=1.0,
    # from_pretrained="PixArt-XL-2-512x512.pth",
    # from_pretrained = "/home/zhaowangbo/wangbo/PixArt-alpha/pretrained_models/OpenSora-v1-HQ-16x512x512.pth",
    # from_pretrained = "OpenSora-v1-HQ-16x512x512.pth",
    from_pretrained="PRETRAINED_MODEL",
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
# mask_ratios = [0.5, 0.29, 0.07, 0.07, 0.07]
# mask_ratios = {
#     "identity": 0.9,
#     "random": 0.06,
#     "mask_head": 0.01,
#     "mask_tail": 0.01,
#     "mask_head_tail": 0.02,
# }
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=120,
    shardformer=True,
)
scheduler = dict(
    type="rflow",
    # timestep_respacing="",
)

# Others
seed = 42
outputs = "outputs"
wandb = True

epochs = 1
log_every = 10
ckpt_every = 1000
load = None

batch_size = 16
lr = 2e-5
grad_clip = 1.0
