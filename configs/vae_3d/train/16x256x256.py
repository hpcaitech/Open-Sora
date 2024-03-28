num_frames = 16
frame_interval = 3
image_size = (256, 256)

# Define dataset
root = None
data_path = "CSV_PATH"
use_image_transform = False
num_workers = 4

# Define acceleration
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model

model = dict(
    type="VAE_3D",
    from_pretrained=None,
    in_out_channels = 4,
    latent_embed_dim = 256,
    filters = 64,
    num_res_blocks = 2,
    channel_multipliers = (1, 2, 2, 4),
    temporal_downsample = (True, True, False),
    num_groups = 32, # for nn.GroupNorm
    conv_downsample = False,
    upsample = "nearest+conv", # options: "deconv", "nearest+conv"
    kl_embed_dim = 64,
    custom_conv_padding = None,
    activation_fn = 'swish',
    kl_weight = 0.000001,
)

# TODO: check all these settings

scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# Others
seed = 42
outputs = "outputs"
wandb = False

epochs = 3
log_every = 1
ckpt_every = 1000
load = None

batch_size = 8
lr = 2e-5
grad_clip = 1.0
