import torch
num_frames = 1
fps = 1
image_size = (256, 256)
dtype = torch.bfloat16
# Define model
model = dict(
    type="DiT-XL/2",
    no_temporal_pos_emb=True,
    condition="label_1000",
    from_pretrained="DiT-XL-2-256x256.pt",
    dtype=dtype
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="classes",
    num_classes=1000,
)
scheduler = dict(
    type="dpm-solver",
    num_sampling_steps=20,
    cfg_scale=4.0,
)

# Others
batch_size = 2
seed = 42
prompt_path = "./assets/texts/imagenet_id.txt"
save_dir = "./samples/samples/"
sample_name = "dit_1x256x256-class"
