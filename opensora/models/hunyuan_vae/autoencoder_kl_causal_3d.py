# Modified from diffusers==0.29.2 and HunyuanVideo
#
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
# Copyright 2024 HunyuanVideo
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config

from opensora.registry import MODELS
from opensora.utils.ckpt import load_checkpoint

try:
    # This diffusers is modified and packed in the mirror.
    from diffusers.loaders import FromOriginalVAEMixin
except ImportError:
    # Use this to be compatible with the original diffusers.
    from diffusers.loaders.single_file_model import FromOriginalModelMixin as FromOriginalVAEMixin

from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook

from opensora.models.hunyuan_vae.vae import (
    DecoderCausal3D,
    DecoderOutput,
    DiagonalGaussianDistribution,
    EncoderCausal3D,
)


@dataclass
class AutoEncoder3DConfig:
    from_pretrained: str | None
    act_fn: str = "silu"
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    layers_per_block: int = 2
    norm_num_groups: int = 32
    scale_factor: float = 0.476986
    shift_factor: float = 0
    time_compression_ratio: int = 4
    spatial_compression_ratio: int = 8
    mid_block_add_attention: bool = True
    block_out_channels: tuple[int] = (128, 256, 512, 512)
    sample_size: int = 256
    sample_tsize: int = 64
    use_slicing: bool = False
    use_spatial_tiling: bool = False
    use_temporal_tiling: bool = False
    tile_overlap_factor: float = 0.25
    dropout: float = 0.0
    channel: bool = False


class AutoencoderKLCausal3D(ModelMixin, ConfigMixin, FromOriginalVAEMixin):
    r"""
    A VAE model with KL loss for encoding images/videos into latents and decoding latent representations into images/videos.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, config: AutoEncoder3DConfig):
        super().__init__()

        self.scale_factor = config.scale_factor
        self.shift_factor = config.shift_factor

        self.time_compression_ratio = config.time_compression_ratio
        self.spatial_compression_ratio = config.spatial_compression_ratio
        self.z_channels = config.latent_channels

        self.encoder = EncoderCausal3D(
            in_channels=config.in_channels,
            out_channels=config.latent_channels,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            act_fn=config.act_fn,
            norm_num_groups=config.norm_num_groups,
            double_z=True,
            time_compression_ratio=config.time_compression_ratio,
            spatial_compression_ratio=config.spatial_compression_ratio,
            mid_block_add_attention=config.mid_block_add_attention,
            dropout=config.dropout,
        )

        self.decoder = DecoderCausal3D(
            in_channels=config.latent_channels,
            out_channels=config.out_channels,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            norm_num_groups=config.norm_num_groups,
            act_fn=config.act_fn,
            time_compression_ratio=config.time_compression_ratio,
            spatial_compression_ratio=config.spatial_compression_ratio,
            mid_block_add_attention=config.mid_block_add_attention,
            dropout=config.dropout,
        )

        self.quant_conv = nn.Conv3d(2 * config.latent_channels, 2 * config.latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(config.latent_channels, config.latent_channels, kernel_size=1)

        self.use_slicing = config.use_slicing
        self.use_spatial_tiling = config.use_spatial_tiling
        self.use_temporal_tiling = config.use_temporal_tiling

        # only relevant if vae tiling is enabled
        self.tile_sample_min_tsize = config.sample_tsize
        self.tile_latent_min_tsize = config.sample_tsize // config.time_compression_ratio

        self.tile_sample_min_size = config.sample_size
        sample_size = config.sample_size[0] if isinstance(config.sample_size, (list, tuple)) else config.sample_size
        self.tile_latent_min_size = int(sample_size / (2 ** (len(config.block_out_channels) - 1)))
        self.tile_overlap_factor = config.tile_overlap_factor

    def enable_temporal_tiling(self, use_tiling: bool = True):
        self.use_temporal_tiling = use_tiling

    def disable_temporal_tiling(self):
        self.enable_temporal_tiling(False)

    def enable_spatial_tiling(self, use_tiling: bool = True):
        self.use_spatial_tiling = use_tiling

    def disable_spatial_tiling(self):
        self.enable_spatial_tiling(False)

    def enable_tiling(self, use_tiling: bool = True):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger videos.
        """
        self.enable_spatial_tiling(use_tiling)
        self.enable_temporal_tiling(use_tiling)

    def disable_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.disable_spatial_tiling()
        self.disable_temporal_tiling()

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]], _remove_lora=False
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor, _remove_lora=_remove_lora)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"), _remove_lora=_remove_lora)

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor, _remove_lora=True)

    @apply_forward_hook
    def encode(
        self,
        x: torch.FloatTensor,
        sample_posterior: bool = True,
        return_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.FloatTensor, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images/videos into latents.

        Args:
            x (`torch.FloatTensor`): Input batch of images/videos.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images/videos. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        assert len(x.shape) == 5, "The input tensor should have 5 dimensions."

        if self.use_temporal_tiling and x.shape[2] > self.tile_sample_min_tsize:
            posterior = self.temporal_tiled_encode(x)
        elif self.use_spatial_tiling and (
            x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size
        ):
            posterior = self.spatial_tiled_encode(x)
        else:
            if self.use_slicing and x.shape[0] > 1:
                encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
                h = torch.cat(encoded_slices)
            else:
                h = self.encoder(x)
            moments = self.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)

        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        z = self.scale_factor * (z - self.shift_factor)  # shift & scale

        if return_posterior:
            return z, posterior
        else:
            return z

    def _decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        assert len(z.shape) == 5, "The input tensor should have 5 dimensions."

        if self.use_temporal_tiling and z.shape[2] > self.tile_latent_min_tsize:
            return self.temporal_tiled_decode(z, return_dict=return_dict)

        if self.use_spatial_tiling and (
            z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size
        ):
            return self.spatial_tiled_decode(z, return_dict=return_dict)

        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z: torch.FloatTensor) -> torch.FloatTensor:
        """
        Decode a batch of images/videos.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        z = z / self.scale_factor + self.shift_factor  # scale & shift

        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample
        return decoded

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
                x / blend_extent
            )
        return b

    def spatial_tiled_encode(self, x: torch.FloatTensor, return_moments: bool = False) -> DiagonalGaussianDistribution:
        r"""Encode a batch of images/videos using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image/videos size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.FloatTensor`): Input batch of images/videos.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoder_kl.AutoencoderKLOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain
                `tuple` is returned.
        """
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split video into tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[-2], overlap_size):
            row = []
            for j in range(0, x.shape[-1], overlap_size):
                tile = x[:, :, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        moments = torch.cat(result_rows, dim=-2)
        if return_moments:
            return moments
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def spatial_tiled_decode(
        self, z: torch.FloatTensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Decode a batch of images/videos using a tiled decoder.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[-2], overlap_size):
            row = []
            for j in range(0, z.shape[-1], overlap_size):
                tile = z[:, :, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=-2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def temporal_tiled_encode(self, x: torch.FloatTensor) -> DiagonalGaussianDistribution:
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_latent_min_tsize - blend_extent

        # Split the video into tiles and encode them separately.
        row = []
        for i in range(0, T, overlap_size):
            tile = x[:, :, i : i + self.tile_sample_min_tsize + 1, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.tile_sample_min_size or tile.shape[-2] > self.tile_sample_min_size
            ):
                tile = self.spatial_tiled_encode(tile, return_moments=True)
            else:
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            row.append(tile)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, : t_limit + 1, :, :])
        moments = torch.cat(result_row, dim=2)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def temporal_tiled_decode(
        self, z: torch.FloatTensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        # Split z into overlapping tiles and decode them separately.

        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_sample_min_tsize - blend_extent

        row = []
        for i in range(0, T, overlap_size):
            tile = z[:, :, i : i + self.tile_latent_min_tsize + 1, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.tile_latent_min_size or tile.shape[-2] > self.tile_latent_min_size
            ):
                decoded = self.spatial_tiled_decode(tile, return_dict=True).sample
            else:
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
            if i > 0:
                decoded = decoded[:, :, 1:, :, :]
            row.append(decoded)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, : t_limit + 1, :, :])

        dec = torch.cat(result_row, dim=2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.FloatTensor, DiagonalGaussianDistribution, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        z, posterior = self.encode(x, return_posterior=True, sample_posterior=sample_posterior, generator=generator)
        dec = self.decode(z)

        return (dec, posterior, z)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query,
        key, value) are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def get_last_layer(self):
        return self.decoder.conv_out.conv.weight

    def get_latent_size(self, input_size: list[int]) -> list[int]:
        latent_size = []
        # T
        latent_size.append((input_size[0] - 1) // self.time_compression_ratio + 1)
        # H, w
        for i in range(1, 3):
            latent_size.append((input_size[i] - 1) // self.spatial_compression_ratio + 1)
        return latent_size


@MODELS.register_module("hunyuan_vae")
def CausalVAE3D_HUNYUAN(
    from_pretrained: str = None,
    device_map: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    **kwargs,
) -> AutoencoderKLCausal3D:
    config = AutoEncoder3DConfig(from_pretrained=from_pretrained, **kwargs)
    with torch.device(device_map):
        model = AutoencoderKLCausal3D(config).to(torch_dtype)
    if from_pretrained:
        model = load_checkpoint(model, from_pretrained, device_map=device_map, strict=True)

    return model
