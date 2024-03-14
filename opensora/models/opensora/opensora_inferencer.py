"""
Adapted from PixArtBaseModel
"""

from typing import Dict, Union

import torch
import torch.nn as nn
from einops import rearrange

from opensora.models.utils import build_module
from opensora.registry import DIFFUSION_SCHEDULERS, MODELS
from opensora.utils.misc import to_torch_dtype

ModelType = Union[Dict, nn.Module]


@MODELS.register_module()
class OpenSoraInferencer(nn.Module):
    """
    Base model for video diffusion;
    Typically, this model consists of:
        - self.vae: image or video VAE
        - self.denoiser: denoiser, typically DiT; currently only support STPixArt
        - self.text_encoder
        - self.scheduler
    TODO:
        - vae/text_encoder/denoiser dtype; do not cast dtype in self.denoiser.forward(); do not pass dtype to self.denoiser
        - self.train_step()
        - only save/load denoiser state_dict
        - DDP inference; currently must be non-DDP
        - support `from_pretrained` for this class?
    """

    def __init__(
        self,
        vae,
        denoiser,
        denoiser_dtype="float32",
        text_encoder=None,
        scheduler=None,
        test_scheduler=None,
    ):
        super().__init__()

        # TODO: currently vae is either AutoencoderKL or AutoencoderKLTemporalDecoder from diffusers
        self.vae = build_module(vae, MODELS)
        self.vae.eval()
        self.vae_dtype = next(self.vae.parameters()).dtype

        self.denoiser = build_module(denoiser, MODELS)
        denoiser_dtype = to_torch_dtype(denoiser_dtype)
        self.denoiser = self.denoiser.to(denoiser_dtype)
        self.denoiser.eval()
        self.denoiser_dtype = denoiser_dtype

        # TODO: currently scheduler is fixed to IDDPM (GaussianDiffusion, SpacedDiffusion)
        self.scheduler = build_module(scheduler, DIFFUSION_SCHEDULERS)
        if test_scheduler is None:
            test_scheduler = scheduler
        self.test_scheduler = build_module(test_scheduler, DIFFUSION_SCHEDULERS)

        # TODO: currently text_encoder is fixed to T5Embedder
        self.text_encoder = build_module(text_encoder, MODELS)

        # self.unet_sample_size = self.denoiser.input_size
        # self.vae_scale_factor = 2 ** (len(self.vae.block_out_channels) - 1)

    @torch.no_grad()
    def inference(
        self,
        prompts,
        height=None,
        width=None,
        latents=None,
        cfg_scale=7.0,
    ):
        """
        Args:
            prompts (List[str]):
        """
        # ======================================================
        # 1. Check inputs & prepare variables
        # ======================================================
        self.check_inputs(prompts, height, width)

        B = len(prompts)
        vae = self.vae.module if hasattr(self.vae, "module") else self.vae
        denoiser = self.denoiser.module if hasattr(self.denoiser, "module") else self.denoiser

        # ======================================================
        # 3. Prepare latent variables (gaussian noise)
        # ======================================================
        if latents is None:
            shape = (B, denoiser.in_channels, *denoiser.input_size)  # concat[B, null]
            latents = torch.randn(shape, device=self.device, dtype=self.denoiser_dtype)
            latents = torch.cat([latents, latents])
        else:
            latents = latents.to(device=self.device, dtype=self.denoiser_dtype)
        # print('loading zw z')
        # latents = torch.load('./checkpoints/debug/z_zw.pth').to(self.device)
        # torch.save(latents.detach().cpu(), './checkpoints/debug/z_pxy.pth')

        # ======================================================
        # 2. Encode input prompt
        # ======================================================
        if self.text_encoder is not None:
            with torch.no_grad():
                caption_embs, emb_masks = self.text_encoder.get_text_embeddings(prompts)
                caption_embs = caption_embs.float()[:, None]
                null_y = denoiser.y_embedder.y_embedding[None].repeat(B, 1, 1)[:, None]
                y = torch.cat([caption_embs, null_y])
        else:
            null_y = denoiser.y_embedder.y_embedding[None].repeat(B, 1, 1)[:, None]
            # y = torch.cat([null_y, null_y])
            y = null_y
            emb_masks = None
        # y = torch.load('./checkpoints/debug/y_zw.pth').to(self.device)
        # emb_masks = torch.load('./checkpoints/debug/emb_mask_zw.pth').to(self.device)
        # print('loading zw y & emb_mask')
        # torch.save(y.detach().cpu(), './checkpoints/debug/y_pxy.pth')  # TODO: a bit difference
        y = y.to(self.denoiser_dtype)

        # ======================================================
        # 4. Reverse process (i.e., denoising loop)
        # ======================================================
        # TODO: refactor scheduler; too ugly
        model_kwargs = {
            "y": y,
            "cfg_scale": cfg_scale,  # MiniSora=7.0
            "mask": emb_masks,
        }
        latents = self.test_scheduler.p_sample_loop(
            denoiser.forward_with_cfg,
            latents.shape,
            latents,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=self.device,
        )  # shape: [2 * B, C, T, H, W]
        torch.save(latents.detach().cpu(), "./checkpoints/debug/latents_pxy.pth")
        latents = latents.chunk(2, dim=0)[0]  # remove null class samples

        # ======================================================
        # 5. Decode latents
        # ======================================================
        B, C, T = latents.shape[:3]
        latents = latents.to(self.vae_dtype)
        with torch.no_grad():
            latents = rearrange(latents, "B C T H W -> (B T) C H W")
            samples = vae.decode(latents / 0.18215, num_frames=T).sample
            samples = rearrange(samples, "(B T) C H W -> B C T H W", B=B)

        return samples

    @property
    def device(self):
        return next(self.denoiser.parameters()).device

    def train(self, mode=True):
        """Set train/eval mode.

        Args:
            mode (bool, optional): Whether set train mode. Defaults to True.
        """
        assert mode is False
        return super().train(mode)
        # if not isinstance(mode, bool):
        #     raise ValueError("training mode is expected to be boolean")
        # self.training = mode
        # self.denoiser.train(mode)
        # # for module in self.denoiser.children():  # modify
        # #     module.train(mode)
        # return self

    def check_inputs(self, prompts, height, width):
        """check whether inputs are in suitable format or not."""

        if not isinstance(prompts, list):
            raise ValueError(f"`prompt` has to be of " f"type `list` but is {type(prompts)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible " f"by 8 but are {height} and {width}.")
