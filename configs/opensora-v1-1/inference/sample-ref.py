num_frames = 16
frame_interval = 3
fps = 24
image_size = (240, 426)
multi_resolution = "STDiT2"

# Condition
prompt_path = None
prompt = None

loop = 10
condition_frame_length = 4
reference_path = [
    "assets/images/condition/cliff.png",
    "assets/images/condition/wave.png",
]
# valid when reference_path is not None
# (loop id, ref id, ref start, length, target start)
mask_strategy = [
    "0,0,0,1,0",
    "0,0,0,1,0",
]

# Define model
model = dict(
    type="STDiT2-XL/2",
    from_pretrained=None,
    input_sq_size=512,
    enable_flashattn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    cache_dir=None,  # "/mnt/hdd/cached_models",
    micro_batch_size=4,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    cache_dir=None,  # "/mnt/hdd/cached_models",
    model_max_length=200,
)
scheduler = dict(
    type="iddpm",
    num_sampling_steps=100,
    cfg_scale=7.0,
    cfg_channel=3,  # or None
)
dtype = "bf16"

# Others
batch_size = 1
seed = 42
save_dir = "./samples/samples/"
