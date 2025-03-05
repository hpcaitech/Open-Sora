_base_ = ["t2i2v_768px.py"]

patch_size = 1
model = dict(
    from_pretrained=None,
    grad_ckpt_settings=None,
    in_channels=512,
)
ae = dict(
    _delete_=True,
    type="dc_ae",
    model_name="dc-ae-f128c512-sana-1.0",
    from_scratch=True,
    from_pretrained="/home/chenli/luchen/Open-Sora-Dev/outputs/250211_114721-vae_train_sana_2d_32channel/epoch13-global_step2000/model/model-00001.safetensors",
)

sampling_option = dict(
    resolution="1024px",
    aspect_ratio="1:1",
    num_frames=16,
    num_steps=50,
    temporal_reduction=1,
    is_causal_vae=False,
    seed=42,
)
fps_save = 24
