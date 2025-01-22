num_frames = 17
image_size = (256, 256)

# Define dataset
dataset = dict(
    type="VideoTextDataset",
    data_path=None,
    num_frames=num_frames,
    frame_interval=1,
    image_size=image_size,
    transform_name="resize_crop",
)

# Define acceleration
num_workers = 16
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"

# Define model
model = dict(
    type="OpenSoraVAE_V1_3",
    from_pretrained=None,
    z_channels=16,
    # use_tiled_conv3d=True,
    # tile_size=16,
)

discriminator = dict(
    type="N_LAYER_DISCRIMINATOR_3D",
    from_pretrained="/mnt/jfs-hdd/sora/checkpoints/pretrained_models/causalvae_v1.1.0/discriminator/model.pt",
    input_nc=3,
    n_layers=3,
    use_actnorm=False,
)

# training
mixed_strategy = "mixed_video_image"
mixed_image_ratio = 0.2

vae_loss_config = dict(perceptual_loss_weight=1.0, kl_loss_weight=1e-6, logvar_init=0.0)

gan_loss_confg = dict(
    disc_start=0,
    disc_factor=1.0,
    disc_weight=0.5,
    disc_loss_type="hinge",
)

# Others
seed = 42
outputs = "outputs/OpenSoraVAE_V1_3"
wandb = True

epochs = 100  # NOTE: adjust accordingly w.r.t dataset size
log_every = 1
ckpt_every = 500
load = None

batch_size = 1
lr = 5e-6
grad_clip = 1.0
warmup_steps = 1000
