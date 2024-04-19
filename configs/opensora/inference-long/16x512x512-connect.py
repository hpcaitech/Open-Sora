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
dtype = "bf16"

# Condition
prompt_path = None
prompt = [
    "Drone view of waves crashing against the rugged cliffs along Big Sur’s garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff’s edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff's edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway.",
    "A sad small cactus with in the Sahara desert becomes happy.",
]

loop = 1
condition_frame_length = 4
reference_path = [
    "assets/images/condition/cliff.png",
    "assets/images/condition/cactus-sad.png;assets/images/condition/cactus-happy.png",
]
mask_strategy = ["0,0,0,1,0;0,0,0,1,-1", "0,0,0,1,0;0,1,0,1,-1"]  # valid when reference_path is not None
# (loop id, ref id, ref start, length, target start)

# Others
batch_size = 2
seed = 42
save_dir = "./samples/samples/"
