num_frames = 33
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

# Define model
model = dict(
    type="OpenSoraVAE_V1_2",
    freeze_vae_2d=False,
    from_pretrained="outputs/vae_stage2",
    cal_loss=True,
)

# loss weights
perceptual_loss_weight = 0.1  # use vgg is not None and more than 0
kl_loss_weight = 1e-6

mixed_strategy = "mixed_video_random"
use_real_rec_loss = True
use_z_rec_loss = False
use_image_identity_loss = False

# Others
seed = 42
outputs = "outputs/vae_stage3"
wandb = False

epochs = 100  # NOTE: adjust accordingly w.r.t dataset size
log_every = 1
ckpt_every = 1000
load = None

batch_size = 1
lr = 1e-5
grad_clip = 1.0
