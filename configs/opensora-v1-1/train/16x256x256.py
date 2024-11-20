# Define dataset
dataset = dict(
    type="VideoTextDataset",
    data_path=None,
    num_frames=16,
    frame_interval=3,
    image_size=(256, 256),
    # t5_offline=True, 
)

mask_ratios = {
    "mask_no": 0.75,
    "mask_quarter_random": 0.025,
    "mask_quarter_head": 0.025,
    "mask_quarter_tail": 0.025,
    "mask_quarter_head_tail": 0.05,
    "mask_image_random": 0.025,
    "mask_image_head": 0.025,
    "mask_image_tail": 0.025,
    "mask_image_head_tail": 0.05,
}

# Define acceleration
num_workers = 4
# dtype = "fp32"
# dtype = "fp16"
dtype = "bf16"
grad_checkpoint = True
# plugin = "ddp"
# plugin = "zero1"
plugin = "zero2-seq"
# sp_size = 1
sp_size = 2
# sp_size = 4
# sp_size = 8

# Define model
model = dict(
    type="STDiT2-XL/2",
    # space_scale=0.5,
    # time_scale=1.0,
    # from_pretrained="./pretrained_models/stdit/OpenSora-STDiT-v2-stage3/model.safetensors",
    # from_pretrained="/home/dist/hpcai/duanjunwen/Open-Sora/outputs/base-STDiT2-XL-2/epoch0-global_step1/",
    input_sq_size=512,  # pretrained model is trained on 512x512
    enable_sequence_parallelism = True,
    qk_norm=True,
    # enable_sequence_parallelism = False, 
    enable_flashattn=False,
    enable_layernorm_kernel=False,
)

vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="./pretrained_models/stabilityai/sd-vae-ft-ema",
)

text_encoder = dict(
    type="t5",
    from_pretrained="./pretrained_models/t5_ckpts/t5-v1_1-xxl",
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

# epochs =  1  
# log_every = 10
# ckpt_every = 1
# load = None

epochs =  20  
log_every = 10
ckpt_every = 3000
load = None

batch_size = 8
lr = 2e-5 # [4e-5, 2e-4], 4e-5 # last 2e-5, 1e-5, 4e-5
grad_clip = 1.0
grad_accm = 4 # world size // sp size
total_steps = 1000

random_dataset = True # set to False, when u use 
benchmark_num_steps = 5
num_ckpt_blocks = 28 # STDIT total 28; bs=16, best 23 or 24; STDIT2 bs=16, best 23 or 24; STDIT2 zero2seq best 5;
cfg_name = "16x256x256"
hidden_dim=1536
# wandb = True