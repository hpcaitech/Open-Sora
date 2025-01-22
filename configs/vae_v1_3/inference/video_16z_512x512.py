image_size = (512, 512)
num_frames = 51

fps = 24
dtype = "bf16"
batch_size = 1
seed = 42
save_dir = "samples/OpenSoraVAE_V1_3_16z"

# Define dataset
dataset = dict(
    type="VideoTextDataset",
    data_path=None,
    num_frames=num_frames,
    image_size=image_size,
    transform_name="resize_crop",
)
num_samples = 100
num_workers = 4
cal_stats = True

# Define model
model = dict(
    type="OpenSoraVAE_V1_3",
    from_pretrained=None,
    z_channels=16,
    micro_batch_size=1,
    micro_batch_size_2d=4,
    micro_frame_size=17,
    use_tiled_conv3d=True,
    tile_size=4,
)
is_init_image = True

vae_loss_config = dict(perceptual_loss_weight=1.0, kl_loss_weight=1e-6, logvar_init=0.0)

gan_loss_confg = dict(
    disc_start=0,
    disc_factor=1.0,
    disc_weight=0.5,
    disc_loss_type="hinge",
)
