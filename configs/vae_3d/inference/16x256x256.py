num_frames = 16
frame_interval = 3
fps = 24 // 3
image_size = (256, 256)
is_vae = True

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
    from_pretrained="PRETRAINED_MODEL", # TODO: double check how this loads the pretrained model
    in_out_channels = 3,
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
)


# Others
seed = 42
save_dir = "outputs/samples"
batch_size = 8

grad_clip = 1.0
