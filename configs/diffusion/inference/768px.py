_base_ = [  # inherit grammer from mmengine
    "256px.py",
    "plugins/sp.py",  # use sequence parallel
]

sampling_option = dict(
    resolution="768px",
)
