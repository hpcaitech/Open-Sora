num_frames = 1
# image_size = (256, 256)
image_size = (1024, 1024)
fps = 24 // 3
max_test_samples = None

# Define dataset
dataset = dict(
    type="VideoTextDataset",
    data_path=None,
    num_frames=num_frames,
    frame_interval=1,
    image_size=image_size,
)

# Define acceleration
num_workers = 4
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1


# Define model
vae_2d = dict(
    type="VideoAutoencoderKL",
    from_pretrained="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
    subfolder="vae",
    micro_batch_size=4,
    local_files_only=True,
)

model = dict(
    type="VAE_Temporal_SD",
)

# discriminator = dict(
#     type="DISCRIMINATOR_3D",
#     image_size=image_size,
#     num_frames=num_frames,
#     in_channels=3,
#     filters=128,
#     channel_multipliers=(2, 4, 4, 4, 4),
#     # channel_multipliers = (2,4,4), #(2,4,4,4,4) # (2,4,4,4) for 64x64 resolution
# )


# loss weights
logvar_init = 0.0
kl_loss_weight = 0.000001
perceptual_loss_weight = 0.1  # use vgg is not None and more than 0
discriminator_factor = 1.0  # for discriminator adversarial loss
# discriminator_loss_weight = 0.5 # for generator adversarial loss
generator_factor = 0.1  # for generator adversarial loss
lecam_loss_weight = None  # NOTE: not clear in MAGVIT what is the weight
discriminator_loss_type = "non-saturating"
generator_loss_type = "non-saturating"
discriminator_start = 2500  # 50000 NOTE: change to correct val, debug use -1 for now
gradient_penalty_loss_weight = None  # 10 # SCH: MAGVIT uses 10, opensora plan doesn't use
ema_decay = 0.999  # ema decay factor for generator


# Others
seed = 42
save_dir = "samples/samples_vae"
wandb = False

# Training
""" NOTE:
magvit uses about # samples (K) * epochs ~ 2-5 K,  num_frames = 4, reso = 128
==> ours num_frams = 16, reso = 256, so samples (K) * epochs ~ [500 - 1200],
3-6 epochs for pexel, from pexel observation its correct
"""


batch_size = 1
lr = 1e-4
grad_clip = 1.0

calc_loss = True
