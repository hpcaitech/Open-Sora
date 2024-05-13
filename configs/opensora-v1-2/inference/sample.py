image_size = (240, 426)
num_frames = 51
fps = 24
frame_interval = 1

prompt_path = "./assets/texts/t2v_sora.txt"
# == Uncomment the following line to use the prompt below ==
# prompt = [
#     'Drone view of waves crashing against the rugged cliffs along Big Sur\'s garay point beach. {"reference_path": "assets/images/condition/cliff.png", "mask_strategy": "0"}',
#     'A breathtaking sunrise scene.{"reference_path": "assets/images/condition/sunset1.png","mask_strategy": "0"}',
#     'A car driving on the ocean.{"reference_path": "https://cdn.openai.com/tmp/s/interp/d0.mp4","mask_strategy": "0,0,-8,0,8"}',
#     'A snowy forest.{"reference_path": "https://cdn.pixabay.com/video/2021/04/25/72171-542991404_large.mp4","mask_strategy": "0,0,0,0,15,0.8"}',
#     'A breathtaking sunrise scene.{"reference_path": "assets/images/condition/sunset1.png;assets/images/condition/sunset2.png","mask_strategy": "0;0,1,0,-1,1"}',
#     '|0|a white jeep equipped with a roof rack driving on a dirt road in a coniferous forest.|2|a white jeep equipped with a roof rack driving on a dirt road in the desert.|4|a white jeep equipped with a roof rack driving on a dirt road in a mountain.|6|A white jeep equipped with a roof rack driving on a dirt road in a city.|8|a white jeep equipped with a roof rack driving on a dirt road on the surface of a river.|10|a white jeep equipped with a roof rack driving on a dirt road under the lake.|12|a white jeep equipped with a roof rack flying into the sky.|14|a white jeep equipped with a roof rack driving in the universe. Earth is the background.{"reference_path": "https://cdn.openai.com/tmp/s/interp/d0.mp4", "mask_strategy": "0,0,0,0,15"}',
# ]

save_dir = "./samples/samples/"
seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"
condition_frame_length = 5
align = 5

model = dict(
    type="STDiT3-XL/2",
    from_pretrained=None,
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderPipeline",
    from_pretrained="pretrained_models/vae-pipeline",
    shift=(-0.10, 0.34, 0.27, 0.98),
    scale=(3.85, 2.32, 2.33, 3.06),
    micro_frame_size=17,
    vae_2d=dict(
        type="VideoAutoencoderKL",
        from_pretrained="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
        subfolder="vae",
        micro_batch_size=4,
        local_files_only=True,
    ),
    vae_temporal=dict(
        type="VAE_Temporal_SD",
        from_pretrained=None,
    ),
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
    local_files_only=True,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)
