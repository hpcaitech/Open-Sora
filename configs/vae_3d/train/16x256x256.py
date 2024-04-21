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

# Define Loss
kl_weight = 0.000001
perceptual_weight = 0.1 # TODO: need to change this to 0.1 !!! according to MAGVIT paper

# Define model

model = dict(
    type="VAE_3D_B",
    in_out_channels = 3,
    latent_embed_dim = 4,
    filters = 64,
    num_res_blocks = 2,
    channel_multipliers = (1, 2, 2, 4),
    temporal_downsample = (True, True, False),
    num_groups = 32, # for nn.GroupNorm
    conv_downsample = False,
    upsample = "nearest+conv", # options: "deconv", "nearest+conv"
    kl_embed_dim = 4,
    custom_conv_padding = None,
    activation_fn = 'swish',
)

# Others
seed = 42
outputs = "outputs"
wandb = False

# Training
''' NOTE: 
magvit uses about # samples (K) * epochs ~ 2-5 K,  num_frames = 4, reso = 128
==> ours num_frams = 16, reso = 256, so samples (K) * epochs ~ [500 - 1200], 
3-6 epochs for pexel, from pexel observation its correct
'''

epochs = 500
log_every = 1
ckpt_every = 200
load = None

batch_size = 4
lr = 1e-4
grad_clip = 1.0
