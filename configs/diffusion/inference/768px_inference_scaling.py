_base_ = [  # inherit grammer from mmengine
    "256px.py",
    "plugins/sp.py",  # use sequence parallel
    "plugins/t2i2v.py",
]
sampling_option = dict(
    resolution="768px",  # 256px or 768px
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