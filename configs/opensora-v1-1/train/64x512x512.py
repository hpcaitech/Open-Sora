# Define dataset
dataset = dict(
    type="VideoTextDataset",
    data_path=None,
    num_frames=64,
    frame_interval=3,
    image_size=(512, 512),
)

# Define acceleration
num_workers = 4
# dtype = "bf16"
dtype = "fp16"
grad_checkpoint = True
plugin = "zero1"
# plugin = "zero2"
# plugin = "zero2-seq"
sp_size = 1
# sp_size = 2
# sp_size = 4
# sp_size = 8

# Define model
model = dict(
    type="STDiT2-XL/2",
    # space_scale=0.5,
    # time_scale=1.0,
    # from_pretrained="./pretrained_models/stdit/OpenSora-STDiT-v2-stage3/model.safetensors",
    # enable_sequence_parallelism = True,
    enable_flashattn=False,
    enable_layernorm_kernel=False,
)
vae = dict(
    type="VideoAutoencoderKL",
    # from_pretrained="stabilityai/sd-vae-ft-ema",
    from_pretrained="./pretrained_models/stabilityai/sd-vae-ft-ema",
    micro_batch_size=64,
)
text_encoder = dict(
    type="t5",
    # from_pretrained="DeepFloyd/t5-v1_1-xxl",
    from_pretrained="./pretrained_models/t5_ckpts/t5-v1_1-xxl",
    model_max_length=200,
    shardformer=True,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# Others
seed = 42
outputs = "outputs"
wandb = False

epochs = 1
log_every = 10
ckpt_every = 250
load = None

# batch_size = 4
# lr = 2e-5
# grad_clip = 1.0

batch_size = 1
lr = 2e-5
grad_clip = 1.0

random_dataset = True
benchmark_num_steps = 4
num_ckpt_blocks = 28 # STDIT total 28; zero2seq best 15;
cfg_name = "64x512x512"
hidden_dim=1536