_base_ = [  # inherit grammer from mmengine
    "768px.py",
    "plugins/t2i2v.py",
]

# # update the inference scaling parameters
sampling_option = dict(
    method="i2v_inference_scaling",
    vbench_dimension_list=['subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality'],
    do_inference_scaling=True,
    num_subtree=3,
    backward_scale=1.0,
    forward_scale=0.5,
    scaling_steps=[1,3,6,10,13],
    vbench_gpus=[6,7],
)