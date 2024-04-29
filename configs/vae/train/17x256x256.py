num_frames = 17
image_size = (256, 256)

# Define dataset
dataset = dict(
    type="VideoTextDataset",
    data_path=None,
    num_frames=num_frames,
    frame_interval=1,
    image_size=image_size,
)

# Define acceleration
num_workers = 16
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# latest
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
#     image_size=image_size,  # NOTE: here image size is different
#     num_frames=num_frames,
#     in_channels=3,
#     filters=128,
#     use_pretrained=True,  # NOTE: set to False only if we want to disable load
#     channel_multipliers=(2, 4, 4, 4, 4),  # (2,4,4,4) for 64x64 resolution
#     # channel_multipliers=(2, 4, 4),  # since on intermediate layer dimension ofs z
# )


# loss weights
logvar_init = 0.0
kl_loss_weight = 0.000001
perceptual_loss_weight = 0.1  # use vgg is not None and more than 0
discriminator_factor = 1.0  # for discriminator adversarial loss
generator_factor = 0.1  # SCH: generator adversarial loss, MAGVIT v2 uses 0.1
lecam_loss_weight = None  # NOTE: MAVGIT v2 use 0.001
discriminator_loss_type = "non-saturating"
generator_loss_type = "non-saturating"
# discriminator_loss_type="hinge"
# generator_loss_type="hinge"
discriminator_start = 2000  # 5000  # 8k data / (8*1) = 1000 steps per epoch
gradient_penalty_loss_weight = None  # 10 # SCH: MAGVIT uses 10, opensora plan doesn't use
ema_decay = 0.999  # ema decay factor for generator


# Others
seed = 42
outputs = "outputs"
wandb = False

# Training
""" NOTE:
magvit uses about # samples (K) * epochs ~ 2-5 K,  num_frames = 4, reso = 128
==> ours num_frams = 16, reso = 256, so samples (K) * epochs ~ [500 - 1200],
3-6 epochs for pexel, from pexel observation its correct
"""

epochs = 100
log_every = 1
ckpt_every = 1000
load = None

batch_size = 4
lr = 1e-5
grad_clip = 1.0
