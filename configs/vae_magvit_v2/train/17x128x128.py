num_frames = 17
frame_interval = 3
image_size = (128, 128)

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
    type="VAE_MAGVIT_V2",
    in_out_channels = 3,
    latent_embed_dim = 256,
    filters = 128,
    num_res_blocks = 4,
    channel_multipliers = (1, 2, 2, 4),
    temporal_downsample = (False, True, True),
    num_groups = 32, # for nn.GroupNorm
    kl_embed_dim = 64,
    activation_fn = 'swish',
    separate_first_frame_encoding = False,
    custom_conv_padding = None,
    # num_frames = num_frames,
    # image_size = image_size,
    # kl_loss_weight = 0.000001,
    # perceptual_loss_weight = 0.1, # use vgg is not None and more than 0
    # discriminator_factor = 1.0,
    # discriminator_in_channels = 3,
    # discriminator_filters = 128,
    # discriminator_channel_multipliers = (2,4,4,4,4),
    # discriminator_loss="hinge",
    # discriminator_start = 50001,
)


discriminator = dict(
    type="DISCRIMINATOR_3D",
    image_size = image_size,
    num_frames = num_frames,
    in_channels = 3,
    filters = 128,
    channel_multipliers = (2,4,4,4,4),
)


# loss weights 
kl_loss_weight = 0.000001,
perceptual_loss_weight = 0.1, # use vgg is not None and more than 0
discriminator_factor = 1.0,
discriminator_loss="hinge",
discriminator_start = 50001,

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

epochs = 10
log_every = 1
ckpt_every = 1000
load = None

batch_size = 4
lr = 1e-4
grad_clip = 1.0
