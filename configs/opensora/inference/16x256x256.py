num_frames = 16
fps = 24 // 3
image_size = (256, 256)

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=0.5,
    time_scale=1.0,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    from_pretrained="PRETRAINED_MODEL",
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    micro_batch_size=4,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=120,
)
scheduler = dict(
    type="iddpm",
    num_sampling_steps=100,
    cfg_scale=7.0,
    cfg_channel=3,  # or None
)
dtype = "bf16"

# Condition
prompt_path = "./assets/texts/t2v_samples.txt"
prompt = None  # prompt has higher priority than prompt_path

# Others
batch_size = 1
seed = 42
save_dir = "./samples/samples/"
