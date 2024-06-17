model = dict(
    type="DBNet",
    backbone=dict(
        type="CLIPResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        style="pytorch",
        dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='https://download.openmmlab.com/mmocr/backbone/resnet50-oclip-7ba0c533.pth'),
        stage_with_dcn=(False, True, True, True),
    ),
    neck=dict(
        type="FPNC",
        in_channels=[256, 512, 1024, 2048],
        lateral_channels=256,
        asf_cfg=dict(attention_type="ScaleChannelSpatial"),
    ),
    det_head=dict(
        type="DBHead",
        in_channels=256,
        module_loss=dict(type="DBModuleLoss"),
        postprocessor=dict(
            type="DBPostprocessor",
            text_repr_type="quad",
            epsilon_ratio=0.002,
        ),
    ),
    data_preprocessor=dict(
        type="TextDetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    init_cfg=dict(
        type="Pretrained",
        checkpoint="https://download.openmmlab.com/mmocr/textdet/dbnetpp/"
        "dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015/"
        "dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac.pth",
    ),
)

test_pipeline = [
    # dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type="Resize", scale=(4068, 1024), keep_ratio=True),
    dict(
        type="PackTextDetInputs",
        # meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'),
        meta_keys=("img_shape", "scale_factor"),
    ),
]

# Visualization
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="TextDetLocalVisualizer",
    name="visualizer",
    vis_backends=vis_backends,
)
