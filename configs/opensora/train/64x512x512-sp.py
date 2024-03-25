num_frames = 64
frame_interval = 2
image_size = (512, 512)

# Define dataset
root = None
data_path = "CSV_PATH"
use_image_transform = False
num_workers = 4

# Define acceleration
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2-seq"
sp_size = 2

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=1.0,
    time_scale=2 / 3,
    from_pretrained=None,
    enable_flashattn=True,
    enable_layernorm_kernel=True,
    enable_sequence_parallelism=True,  # enable sq here
)
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

batch_size = 1
lr = 2e-5
grad_clip = 1.0
