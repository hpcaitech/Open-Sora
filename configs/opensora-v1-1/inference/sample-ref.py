num_frames = 16
frame_interval = 3
fps = 24
image_size = (240, 426)
multi_resolution = "STDiT2"

# Condition
prompt_path = None
prompt = [
    "A car driving on the ocean.",
    'Drone view of waves crashing against the rugged cliffs along Big Sur\'s garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff\'s edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff\'s edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway.{"reference_path": "assets/images/condition/cliff.png", "mask_strategy": "0"}',
    "In an ornate, historical hall, a massive tidal wave peaks and begins to crash. Two surfers, seizing the moment, skillfully navigate the face of the wave.",
]

loop = 2
condition_frame_length = 4
reference_path = [
    "https://cdn.openai.com/tmp/s/interp/d0.mp4",
    None,
    "assets/images/condition/wave.png",
]
# valid when reference_path is not None
# (loop id, ref id, ref start, length, target start)
mask_strategy = [
    "0,0,0,0,8,0.3",
    None,
    "0",
]

# Define model
model = dict(
    type="STDiT2-XL/2",
    from_pretrained=None,
    input_sq_size=512,
    qk_norm=True,
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
