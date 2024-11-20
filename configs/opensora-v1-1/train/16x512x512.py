# Define dataset
dataset = dict(
    type="VideoTextDataset",
    data_path=None,
    num_frames=16,
    frame_interval=3,
    image_size=(512, 512),
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
dtype = "bf16"
grad_checkpoint = True
# plugin = "ddp"
# plugin = "zero2"
plugin = "zero2-seq"
# sp_size = 1
# sp_size = 2
sp_size = 4
# sp_size = 8

# Define model
model = dict(
    type="STDiT2-XL/2",
    # space_scale=0.5,
    # time_scale=1.0,
    # from_pretrained="./pretrained_models/stdit/OpenSora-STDiT-v2-stage3/model.safetensors",
    input_sq_size=512,  # pretrained model is trained on 512x512
    enable_sequence_parallelism = True, 
    enable_flashattn=False,
    enable_layernorm_kernel=False,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="./pretrained_models/stabilityai/sd-vae-ft-ema",
    micro_batch_size=128,
)
text_encoder = dict(
    type="t5",
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
ckpt_every = 100
load = None


batch_size = 4
lr = 2e-5
grad_clip = 1.0

random_dataset = True
benchmark_num_steps = 5
num_ckpt_blocks = 21 # STDIT total 28
cfg_name = "16x512x512"
hidden_dim=1536