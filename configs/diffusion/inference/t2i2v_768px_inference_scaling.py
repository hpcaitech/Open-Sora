_base_ = [  # inherit grammer from mmengine
    "768px.py",
    "plugins/t2i2v.py",
]

# # update the inference scaling parameters
# sampling_option = dict(
#     method="i2v_inference_scaling",
#     vbench_dimension_list=['subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality'],
#     do_inference_scaling=True,
#     num_subtree=3,
#     backward_scale=1.0,
#     forward_scale=0.5,
#     scaling_steps=[1,2,4,8,13],
#     vbench_gpus=[4,5,6,7],
#     seed=42
# )

# second setting
sampling_option = dict(
    method="i2v_inference_scaling",
    vbench_dimension_list=['subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality'],
    do_inference_scaling=True,
    num_subtree=5,
    backward_scale=1.0,
    forward_scale=0.5,
    scaling_steps=[1,2,3,4,6,8,10,13],
    vbench_gpus=[6,7],
    seed=42
)

# third setting
# sampling_option = dict(
#     method="i2v_inference_scaling",
#     vbench_dimension_list=['subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality'],
#     do_inference_scaling=True,
#     num_subtree=8,
#     backward_scale=0.78,
#     forward_scale=0.83,
#     scaling_steps=[1,2,3,4,5,6,7,8,9,12,15,18,20,22,25,30,35],
#     vbench_gpus=[6,7],
#     seed=42
# )
