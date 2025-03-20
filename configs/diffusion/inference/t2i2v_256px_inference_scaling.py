_base_ = [  # inherit grammer from mmengine
    "256px.py",
    "plugins/t2i2v.py",
]

# update the inference scaling parameters
sampling_option = dict(
    method="i2v_inference_scaling",
    vbench_dimension_list=['subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality'],
    do_inference_scaling=True,
    num_subtree=3,
    backward_scale=1.0,
    forward_scale=0.5,
    scaling_steps=[1,2,4,7,9,15,20],
    vbench_gpus=[4,5,6,7],
)