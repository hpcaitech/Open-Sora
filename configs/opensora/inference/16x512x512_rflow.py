num_frames = 16
fps = 24 // 3
image_size = (512, 512)

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    enable_flashattn=True,
    enable_layernorm_kernel=True,
    # from_pretrained="PRETRAINED_MODEL",
    from_pretrained="./pretrained_models/stdit/OpenSora/OpenSora-v1-HQ-16x512x512.pth",
)
vae = dict(
    type="VideoAutoencoderKL",
    # from_pretrained="stabilityai/sd-vae-ft-ema",
    from_pretrained="./pretrained_models/stabilityai/sd-vae-ft-ema",
    micro_batch_size=2,
)
text_encoder = dict(
    type="t5",
    # from_pretrained="DeepFloyd/t5-v1_1-xxl",
    from_pretrained="./pretrained_models/t5_ckpts/t5-v1_1-xxl",
    model_max_length=120,
)
scheduler = dict(
    type="rflow",
    num_sampling_steps=10,
    cfg_scale=7.0,
)
dtype = "bf16"

# Others
batch_size = 2
seed = 42
prompt_path = "./assets/texts/t2v_samples.txt"
save_dir = "./outputs/samples3/"