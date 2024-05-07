import torch

num_frames = 16
fps = 8
image_size = (256, 256)

dtype = torch.bfloat16
# Define model
model = dict(
    type="Latte-XL/2",
    condition="label_101",
    from_pretrained="Latte-XL-2-256x256-ucf101.pt",
    dtype=dtype
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="classes",
    num_classes=101,
)
scheduler = dict(
    type="dpm-solver",
    num_sampling_steps=20,
    cfg_scale=4.0,
)


# Others
batch_size = 2
seed = 42
prompt_path = "./assets/texts/ucf101_id.txt"
sample_name = "latte_ucf101_id"
save_dir = "./samples/samples/"
