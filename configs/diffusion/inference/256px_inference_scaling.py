_base_ = [  # inherit grammer from mmengine
    "256px.py",
    "plugins/t2i2v.py",
    "plugins/tp.py",  # use tensor parallel
]
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
    method="i2v_inference_scaling",  # hard-coded for now
    vbench_dimension_list=['subject_consistency'],
    do_inference_scaling=True,
    num_subtree=3,
    backward_scale=0.78,
    forward_scale=0.83,
    scaling_steps=[1,2,4,7,9,15,20],
    seed=None,  # random seed for z
    vbench_gpus=[4,5,6,7]
)