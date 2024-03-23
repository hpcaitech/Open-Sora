# scripts/inference_long.py
num_frames = 16
fps = 24 // 3
image_size = (512, 512)

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    use_x_mask=True,
    enable_flashattn=True,
    enable_layernorm_kernel=True,
    from_pretrained=None,
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
    # type="dpm-solver",
    num_sampling_steps=100,
    cfg_scale=7.0,
)
dtype = "fp16"

# Condition
prompt_path = None
prompt = [
    "In an ornate, historical hall, a massive tidal wave peaks and begins to crash. Two surfers, seizing the moment, skillfully navigate the face of the wave."
]

loop = 5
condition_frame_length = 4
reference_path = ["assets/images/condition/wave.png"]
mask_strategy = ["0,0,0,1,0"]  # valid when reference_path is not None
# (loop id, ref id, ref start, length, target start)

# Others
batch_size = 2
seed = 42
save_dir = "./outputs/samples/"
