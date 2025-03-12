save_dir = "samples"  # save directory
seed = 42  # random seed (except seed for z)
batch_size = 1
dtype = "bf16"

cond_type = "t2v"
# conditional inference options:
# t2v: text-to-video
# i2v_head: image-to-video (head)
# i2v_tail: image-to-video (tail)
# i2v_loop: connect images
# v2v_head_half: video extension with first half
# v2v_tail_half: video extension with second half

dataset = dict(type="text")
sampling_option = dict(
    resolution="256px",  # 256px or 768px
    aspect_ratio="16:9",  # 9:16 or 16:9 or 1:1
    num_frames=129,  # number of frames
    num_steps=50,  # number of steps
    shift=True,
    temporal_reduction=4,
    is_causal_vae=True,
    guidance=7.5,  # guidance for text-to-video
    guidance_img=3.0,  # guidance for image-to-video
    text_osci=True,  # enable text guidance oscillation
    image_osci=True,  # enable image guidance oscillation
    scale_temporal_osci=True,
    method="i2v",  # hard-coded for now
    seed=None,  # random seed for z
)
motion_score = "4"  # motion score for video generation
fps_save = 24  # fps for video generation and saving

# Define model components
model = dict(
    type="flux",
    from_pretrained="./ckpts/Open_Sora_v2.safetensors",
    guidance_embed=False,
    fused_qkv=False,
    use_liger_rope=True,
    # model architecture
    in_channels=64,
    vec_in_dim=768,
    context_in_dim=4096,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    cond_embed=True,
)
ae = dict(
    type="hunyuan_vae",
    from_pretrained="./ckpts/hunyuan_vae.safetensors",
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,
    use_spatial_tiling=True,
    use_temporal_tiling=False,
)
t5 = dict(
    type="text_embedder",
    from_pretrained="./ckpts/google/t5-v1_1-xxl",
    max_length=512,
    shardformer=True,
)
clip = dict(
    type="text_embedder",
    from_pretrained="./ckpts/openai/clip-vit-large-patch14",
    max_length=77,
)
