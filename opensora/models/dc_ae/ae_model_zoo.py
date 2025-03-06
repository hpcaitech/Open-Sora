# Copyright 2024 MIT Han Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

import diffusers
import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from opensora.registry import MODELS
from opensora.utils.ckpt import load_checkpoint

from .models.dc_ae import DCAE, DCAEConfig, dc_ae_f32, dc_ae_f64c128, dc_ae_f64t4c256, dc_ae_f128c512

__all__ = ["create_dc_ae_model_cfg", "DCAE_HF", "AutoencoderKL", "DC_AE"]


REGISTERED_DCAE_MODEL: dict[str, tuple[Callable, Optional[str]]] = {
    "dc-ae-f32c32-in-1.0": (dc_ae_f32, None),
    "dc-ae-f64c128-in-1.0": (dc_ae_f64c128, None),
    "dc-ae-f128c512-in-1.0": (dc_ae_f128c512, None),
    #################################################################################################
    "dc-ae-f32c32-mix-1.0": (dc_ae_f32, None),
    "dc-ae-f64c128-mix-1.0": (dc_ae_f64c128, None),
    "dc-ae-f128c512-mix-1.0": (dc_ae_f128c512, None),
    #################################################################################################
    "dc-ae-f32c32-sana-1.0": (dc_ae_f32, None),
    "dc-ae-f32c32-sana-1.0-video": (dc_ae_f32, None),
    "dc-ae-f32c32-sana-1.0-video-temporal-compression": (dc_ae_f32, None),
    "dc-ae-f32t4c256": (dc_ae_f32, None),
    "dc-ae-f32t4c128": (dc_ae_f32, None),
    "dc-ae-f32t4c64": (dc_ae_f32, None),
    "dc-ae-f64t4c128": (dc_ae_f64c128, None),
    "dc-ae-f64t4c256": (dc_ae_f64t4c256, None),
}


def create_dc_ae_model_cfg(name: str, pretrained_path: Optional[str] = None) -> DCAEConfig:
    assert name in REGISTERED_DCAE_MODEL, f"{name} is not supported"
    dc_ae_cls, default_pt_path = REGISTERED_DCAE_MODEL[name]
    pretrained_path = default_pt_path if pretrained_path is None else pretrained_path
    model_cfg = dc_ae_cls(name, pretrained_path)
    return model_cfg


class DCAE_HF(DCAE, PyTorchModelHubMixin):
    def __init__(self, model_name: str):
        cfg = create_dc_ae_model_cfg(model_name)
        DCAE.__init__(self, cfg)


@MODELS.register_module("dc_ae")
def DC_AE(
    model_name: str,
    device_map: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    from_scratch: bool = False,
    from_pretrained: str | None = None,
    is_training: bool = False,
    rename_keys: dict = None,
    use_spatial_tiling: bool = False,
    use_temporal_tiling: bool = False,
    spatial_tile_size: int = 256,
    temporal_tile_size: int = 32,
    tile_overlap_factor: float = 0.25,
    tune_channel_proj: bool = False,
    train_decoder_only: bool = False,
    scaling_factor: float = None,
) -> DCAE_HF:
    if not from_scratch:
        model = DCAE_HF.from_pretrained(model_name).to(device_map, torch_dtype)
    else:
        model = DCAE_HF(model_name).to(device_map, torch_dtype)

    if from_pretrained is not None:
        model = load_checkpoint(model, from_pretrained, device_map=device_map, rename_keys=rename_keys)
        print(f"loaded dc_ae from ckpt path: {from_pretrained}")

    if train_decoder_only:
        print("freezing all encoder weights!")
    # tune the channel projection layer only
    if tune_channel_proj:
        model.cfg.tune_channel_proj = True
        print("freezing all weights except the channel projection!")
        for _, param in model.named_parameters():
            param.requires_grad = False
        for _, param in model.decoder.named_parameters():
            param.requires_grad = True

    model.cfg.is_training = is_training
    model.use_spatial_tiling = use_spatial_tiling
    model.use_temporal_tiling = use_temporal_tiling
    model.spatial_tile_size = spatial_tile_size
    model.temporal_tile_size = temporal_tile_size
    model.tile_overlap_factor = tile_overlap_factor
    if scaling_factor is not None:
        model.scaling_factor = scaling_factor
    return model


class AutoencoderKL(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        if self.model_name in ["stabilityai/sd-vae-ft-ema"]:
            self.model = diffusers.models.AutoencoderKL.from_pretrained(self.model_name)
            self.spatial_compression_ratio = 8
        elif self.model_name == "flux-vae":
            from diffusers import FluxPipeline

            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
            self.model = diffusers.models.AutoencoderKL.from_pretrained(pipe.vae.config._name_or_path)
            self.spatial_compression_ratio = 8
        else:
            raise ValueError(f"{self.model_name} is not supported for AutoencoderKL")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_name in ["stabilityai/sd-vae-ft-ema", "flux-vae"]:
            return self.model.encode(x).latent_dist.sample()
        else:
            raise ValueError(f"{self.model_name} is not supported for AutoencoderKL")

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.model_name in ["stabilityai/sd-vae-ft-ema", "flux-vae"]:
            return self.model.decode(latent).sample
        else:
            raise ValueError(f"{self.model_name} is not supported for AutoencoderKL")
