num_frames = 64
fps = 24 // 2
image_size = (512, 512)

# model config
model = dict(
    type="STDiT-XL/2",
    space_scale=1.0,
    time_scale=2 / 3,
    from_pretrained="PRETRAINED_MODEL",
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="t5",
    from_pretrained="./pretrained_models/t5_ckpts",
    model_max_length=120,
)
scheduler = dict(
    type="iddpm",
    num_sampling_steps=100,
    cfg_scale=7.0,
)
dtype = "fp16"

batch_size = 1
seed = 42
prompt_path = "./assets/texts/t2v_samples.txt"
save_dir = "./samples"
