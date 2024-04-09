# Define dataset
dataset = dict(
    type="VariableVideoTextDataset",
    data_path=None,
    num_frames=None,
    frame_interval=1,
    image_size=(None, None),
    transform_name="resize_crop",
)
bucket_config = {
    # "240p": {128: (1.0, 2)}, # 4.28s/it
    # "240p": {64: (1.0, 4)},
    # "240p": {32: (1.0, 8)},  # 4.6s/it
    # "240p": {16: (1.0, 16)},  # 4.6s/it
    # "480p": {16: (1.0, 4)},  # 4.6s/it
    "720p": {16: (1.0, 2)},  # 5.89s/it
    # "256": {1: (1.0, 256)},  # 4.5s/it
    # "512": {1: (1.0, 96)}, # 4.7s/it
    # "512": {1: (1.0, 128)}, # 6.3s/it
    # "480p": {1: (1.0, 50)},  # 4.0s/it
    # "1024": {1: (1.0, 32)},  # 6.8s/it
    # "1024": {1: (1.0, 20)}, # 4.3s/it
    # "1080p": {1: (1.0, 16)}, # 8.6s/it
    # "1080p": {1: (1.0, 8)},  # 4.4s/it
}
# mask_ratios = {
#     "mask_no": 0.0,
#     "mask_random": 1.0,
# }

# Define acceleration
num_workers = 4
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="STDiT2-XL/2",
    from_pretrained=None,
    input_sq_size=512,  # pretrained model is trained on 512x512
    enable_flashattn=True,
    enable_layernorm_kernel=True,
)
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
    type="iddpm-speed",
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

batch_size = 10  # only for logging
lr = 2e-5
grad_clip = 1.0
