import torch

num_frames = 16
denoiser_dtype = torch.float16

model = dict(
    type="OpenSoraInferencer",
    denoiser_dtype=denoiser_dtype,
    vae=dict(
        type="DFSAutoencoderKLTemporalDecoder",
        from_pretrained="./checkpoints/pretrained_models/vae_temporal_decoder",
    ),
    denoiser=dict(
        type="OpenSora",
        input_size=(num_frames, 32, 32),  # [T, H, W] / vae.[T, H, W]
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        space_scale=0.5,
        dtype=denoiser_dtype,  # TODO: do not pass dtype here
    ),
    # scheduler=dict(
    #     type='IDDPM',
    #     timestep_respacing=None,
    #     learn_sigma=True,
    #     pred_sigma=True,
    #     snr=False,
    # ),
    scheduler=dict(
        type="IDDPMDiT",
        timestep_respacing=None,
    ),
    # text_encoder=None,
    text_encoder=dict(
        type="T5Embedder",
        device="cuda",
        local_cache=True,
        cache_dir="./checkpoints/pretrained_models/t5_ckpts",
        torch_dtype=torch.float,
    ),
)

ckpt_path = "./checkpoints/outputs/285-F16S3-PixArt-ST-XL-2/epoch512-global_step20000/model_ckpt.pt"

del torch
del num_frames
del denoiser_dtype
