# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse

from colossalai.utils import get_current_device
from torchvision.io import write_video
from transformers import AutoModel, AutoTokenizer, CLIPTextModel

from open_sora.diffusion import create_diffusion
from open_sora.modeling import DiT_models
from open_sora.utils.data import col2video, unnormalize_video


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = get_current_device()
    if len(args.vqvae) > 0:
        vqvae = (
            AutoModel.from_pretrained(args.vqvae, trust_remote_code=True)
            .to(device)
            .eval()
        )
        in_channels = vqvae.embedding_dim
        w_h_factor = 4
        t_factor = 2
    else:
        # disable VQ-VAE if not provided, just use raw video frames
        vqvae = None
        in_channels = 3
        w_h_factor = 1
        t_factor = 1
    text_model = CLIPTextModel.from_pretrained(args.text_model).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)

    model = DiT_models[args.model](in_channels=in_channels).to(device).eval()
    patch_size = model.patch_size
    model.load_state_dict(torch.load(args.ckpt))
    diffusion = create_diffusion(str(args.num_sampling_steps))

    # Create sampling noise:
    text_inputs = tokenizer(args.text, return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_latent_states = text_model(**text_inputs).last_hidden_state

    num_frames = args.fps * args.sec
    z = torch.randn(
        1,
        (args.height // patch_size // w_h_factor)
        * (args.width // patch_size // w_h_factor)
        * (num_frames // t_factor),
        in_channels,
        patch_size,
        patch_size,
        device=device,
    )

    # Setup classifier-free guidance:
    model_kwargs = {}
    z = torch.cat([z, z], 0)
    model_kwargs["text_latent_states"] = torch.cat(
        [text_latent_states, torch.zeros_like(text_latent_states)], 0
    )
    model_kwargs["cfg_scale"] = args.cfg_scale
    model_kwargs["attention_mask"] = torch.ones(
        2, 1, z.shape[1], text_latent_states.shape[1], device=device, dtype=torch.int
    )

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = col2video(
        samples.squeeze(),
        (
            num_frames // t_factor,
            in_channels,
            args.height // w_h_factor,
            args.width // w_h_factor,
        ),
    )
    if vqvae is not None:
        # [T, C, H, W] -> [B, C, T, H, W]
        samples = samples.permute(1, 0, 2, 3).unsqueeze(0)
        samples = vqvae.decode_from_embeddings(samples)
        # [B, C, T, H, W] -> [T, H, W, C]
        samples = samples.squeeze(0).permute(1, 2, 3, 0)
    else:
        # [T, C, H, W] -> [T, H, W, C]
        samples = samples.permute(0, 2, 3, 1)
    samples = unnormalize_video(samples).to(torch.uint8)

    write_video("sample.mp4", samples.cpu(), args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/8"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="a cartoon animals runs through an ice cave in a video game",
    )
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).",
    )
    parser.add_argument("--vqvae", default="hpcai-tech/vqvae")
    parser.add_argument(
        "--text_model", type=str, default="openai/clip-vit-base-patch32"
    )
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--sec", type=int, default=8)
    args = parser.parse_args()
    main(args)
