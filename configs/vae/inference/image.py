image_size = (256, 256)
num_frames = 1

dtype = "bf16"
batch_size = 1
seed = 42
save_dir = "samples/vae_video"
cal_stats = True
log_stats_every = 100

# Define dataset
dataset = dict(
    type="VideoTextDataset",
    data_path=None,
    num_frames=num_frames,
    image_size=image_size,
)
num_samples = 100
num_workers = 4

# Define model
model = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
    micro_frame_size=None,
    micro_batch_size=4,
    cal_loss=True,
)

# loss weights
perceptual_loss_weight = 0.1  # use vgg is not None and more than 0
kl_loss_weight = 1e-6
