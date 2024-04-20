# scripts/inference_long.py
num_frames = 16
fps = 24 // 3
image_size = (256, 256)

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=0.5,
    time_scale=1.0,
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
    num_sampling_steps=100,
    cfg_scale=7.0,
)
dtype = "fp16"

# Condition
prompt_path = None
prompt = [
    "A car driving on a road in the middle of a desert.",
]

loop = 1
condition_frame_length = 4
reference_path = [
    "https://cdn.openai.com/tmp/s/interp/d0.mp4",
]
mask_strategy = [
    "0,0,0,1,0,0",
]  # valid when reference_path is not None
# (loop id, ref id, ref start, length, target start)

# Others
batch_size = len(prompt)
seed = 42
save_dir = "outputs/SDEdit"
