num_frames = 1
fps = 24 // 3
# image_size = (1358, 680)
image_size = (2160, 3840)
multi_resolution = "STDiT2"

# Define model
model = dict(
    type="STDiT2-XL/2",
    from_pretrained="PixArt-XL-2-1024-MS.pth",
    input_sq_size=512,
    enable_flashattn=True,
    enable_layernorm_kernel=True,
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
dtype = "fp16"

# Condition
prompt_path = "./assets/texts/t2v_samples.txt"
prompt = None  # prompt has higher priority than prompt_path

# Others
batch_size = 1
seed = 42
save_dir = "./outputs/samples/"
