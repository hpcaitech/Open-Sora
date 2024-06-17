num_frames = 16
frame_interval = 3
fps = 24
image_size = (240, 426)
multi_resolution = "STDiT2"

# Condition
prompt_path = None
prompt = [
    'Drone view of waves crashing against the rugged cliffs along Big Sur\'s garay point beach. {"reference_path": "assets/images/condition/cliff.png", "mask_strategy": "0"}',
    'A breathtaking sunrise scene.{"reference_path": "assets/images/condition/sunset1.png","mask_strategy": "0"}',
    'A car driving on the ocean.{"reference_path": "https://cdn.openai.com/tmp/s/interp/d0.mp4","mask_strategy": "0,0,-8,0,8"}',
    'A snowy forest.{"reference_path": "https://cdn.pixabay.com/video/2021/04/25/72171-542991404_large.mp4","mask_strategy": "0,0,0,0,15,0.8"}',
    'A breathtaking sunrise scene.{"reference_path": "assets/images/condition/sunset1.png;assets/images/condition/sunset2.png","mask_strategy": "0;0,1,0,-1,1"}',
    '|0|a white jeep equipped with a roof rack driving on a dirt road in a coniferous forest.|2|a white jeep equipped with a roof rack driving on a dirt road in the desert.|4|a white jeep equipped with a roof rack driving on a dirt road in a mountain.|6|A white jeep equipped with a roof rack driving on a dirt road in a city.|8|a white jeep equipped with a roof rack driving on a dirt road on the surface of a river.|10|a white jeep equipped with a roof rack driving on a dirt road under the lake.|12|a white jeep equipped with a roof rack flying into the sky.|14|a white jeep equipped with a roof rack driving in the universe. Earth is the background.{"reference_path": "https://cdn.openai.com/tmp/s/interp/d0.mp4", "mask_strategy": "0,0,0,0,15"}',
]

loop = 2
condition_frame_length = 4
# (
#   loop id, [the loop index of the condition image or video]
#   reference id, [the index of the condition image or video in the reference_path]
#   reference start, [the start frame of the condition image or video]
#   target start, [the location to insert]
#   length, [the number of frames to insert]
#   edit_ratio [the edit rate of the condition image or video]
# )
# See https://github.com/hpcaitech/Open-Sora/blob/main/docs/config.md#advanced-inference-config for more details
# See https://github.com/hpcaitech/Open-Sora/blob/main/docs/commands.md#inference-with-open-sora-11 for more examples

# Define model
model = dict(
    type="STDiT2-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v2-stage3",
    input_sq_size=512,
    qk_norm=True,
    qk_norm_legacy=True,
    enable_flash_attn=True,
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
