num_frames = 1
frame_interval = 1
fps = 24
image_size = (256, 256)

# Define dataset
dataset = dict(
    type="VideoTextDataset",
    data_path=None,
    num_frames=num_frames,
    frame_interval=1,
    image_size=image_size,
)
num_workers = 4
max_test_samples = None

# Define model
model = dict(
    type="VideoAutoencoderPipeline",
    freeze_vae_2d=True,
    vae_2d=dict(
        type="VideoAutoencoderKL",
        from_pretrained="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
        subfolder="vae",
        micro_batch_size=4,
        local_files_only=True,
    ),
    vae_temporal=dict(
        type="VAE_Temporal_SD",
        from_pretrained=None,
    ),
)
dtype = "bf16"

# loss weights
perceptual_loss_weight = 0.1  # use vgg is not None and more than 0
kl_loss_weight = 1e-6

# Others
batch_size = 1
seed = 42
save_dir = "samples/vae_image"
