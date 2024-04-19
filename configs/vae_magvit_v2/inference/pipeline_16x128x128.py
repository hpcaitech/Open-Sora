num_frames = 16
frame_interval = 3
image_size = (128, 128)
fps = 24 // 3
is_vae = True
use_pipeline = True

# Define dataset
root = None
data_path = "CSV_PATH"
max_test_samples = -1
use_image_transform = False
num_workers = 4
video_contains_first_frame = False



# Define acceleration
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1


# Define model

vae_2d = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)

model = dict(
    type="VAE_MAGVIT_V2",
    in_out_channels = 4,
    latent_embed_dim = 256,
    filters = 128,
    num_res_blocks = 4,
    channel_multipliers = (1, 2, 2, 4),
    temporal_downsample = (False, True, True),
    num_groups = 32, # for nn.GroupNorm
    kl_embed_dim = 64,
    activation_fn = 'swish',
    separate_first_frame_encoding = False,
    disable_space = True,
    custom_conv_padding = None
)


discriminator = dict(
    type="DISCRIMINATOR_3D",
    image_size = 16,
    num_frames = num_frames,
    in_channels = 4,
    filters = 128,
    channel_multipliers = (2,4,4), #(2,4,4,4,4) # (2,4,4,4) for 64x64 resolution
)


# loss weights 
logvar_init=0.0
kl_loss_weight = 0.000001
perceptual_loss_weight = 0.1 # use vgg is not None and more than 0
discriminator_factor = 1.0 # for discriminator adversarial loss
# discriminator_loss_weight = 0.5 # for generator adversarial loss
generator_factor = 0.1 # for generator adversarial loss
lecam_loss_weight = None # NOTE: not clear in MAGVIT what is the weight
discriminator_loss_type="non-saturating"
generator_loss_type="non-saturating"
discriminator_start = 2500 # 50000 NOTE: change to correct val, debug use -1 for now
gradient_penalty_loss_weight = None # 10 # SCH: MAGVIT uses 10, opensora plan doesn't use
ema_decay = 0.999  # ema decay factor for generator


# Others
seed = 42
save_dir = "outputs/pipeline_samples_v2"
wandb = False

# Training
''' NOTE: 
magvit uses about # samples (K) * epochs ~ 2-5 K,  num_frames = 4, reso = 128
==> ours num_frams = 16, reso = 256, so samples (K) * epochs ~ [500 - 1200], 
3-6 epochs for pexel, from pexel observation its correct
'''

epochs = 10
log_every = 1
ckpt_every = 500
load = None

batch_size = 4
lr = 1e-4
grad_clip = 1.0

calc_loss = True