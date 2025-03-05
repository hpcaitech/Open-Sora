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
# # 
# Copyright 2024 HunyuanVideo
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import Attention
from diffusers.utils import logging
from einops import rearrange
from torch import nn

from opensora.acceleration.checkpoint import auto_grad_checkpoint
from opensora.models.vae.utils import ChannelChunkConv3d, get_conv3d_n_chunks

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

INTERPOLATE_NUMEL_LIMIT = 2**31 - 1


def chunk_nearest_interpolate(
    x: torch.Tensor,
    scale_factor,
):
    limit = INTERPOLATE_NUMEL_LIMIT // np.prod(scale_factor)
    n_chunks = get_conv3d_n_chunks(x.numel(), x.size(1), limit)
    x_chunks = x.chunk(n_chunks, dim=1)
    x_chunks = [F.interpolate(x_chunk, scale_factor=scale_factor, mode="nearest") for x_chunk in x_chunks]
    return torch.cat(x_chunks, dim=1)


def prepare_causal_attention_mask(n_frame: int, n_hw: int, dtype, device, batch_size: int = None):
    seq_len = n_frame * n_hw
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    for i in range(seq_len):
        i_frame = i // n_hw
        mask[i, : (i_frame + 1) * n_hw] = 0
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class CausalConv3d(nn.Module):
    """
    Implements a causal 3D convolution layer where each position only depends on previous timesteps and current spatial locations.
    This maintains temporal causality in video generation tasks.
    """

    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        pad_mode="replicate",
        **kwargs,
    ):
        super().__init__()

        self.pad_mode = pad_mode
        padding = (
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size - 1,
            0,
        )  # W, H, T
        self.time_causal_padding = padding

        self.conv = ChannelChunkConv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class ChannelDuplicatingPixelShuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        factor=(1, 2, 2),
        slice_t=False,  # either slice T or pad T
    ):
        super().__init__()
        self.factor = factor
        self.slice_t = slice_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(2)
        if self.factor[0] == 2:
            if T == 1:  # image
                x = x.repeat_interleave(self.factor[1] * self.factor[2], dim=1)
                residual = rearrange(
                    x, "B (C fh fw) T H W -> B C T (H fh) (W fw)", fh=self.factor[1], fw=self.factor[2]
                )
            else:  # video
                if self.slice_t:
                    # slice T and process differently
                    first_f, other_f = x.split((1, T - 1), dim=2)
                    first_f = first_f.repeat_interleave(self.factor[1] * self.factor[2], dim=1)
                    first_f = rearrange(
                        first_f, "B (C fh fw) T H W -> B C T (H fh) (W fw)", fh=self.factor[1], fw=self.factor[2]
                    )
                    other_f = other_f.repeat_interleave(self.factor[0] * self.factor[1] * self.factor[2], dim=1)
                    other_f = rearrange(
                        other_f,
                        "B (C ft fh fw) T H W -> B C (T ft) (H fh) (W fw)",
                        ft=self.factor[0],
                        fh=self.factor[1],
                        fw=self.factor[2],
                    )
                    residual = torch.cat((first_f, other_f), dim=2)
                else:
                    x = x.repeat_interleave(self.factor[0] * self.factor[1] * self.factor[2], dim=1)
                    residual = rearrange(
                        x,
                        "B (C ft fh fw) T H W -> B C (T ft) (H fh) (W fw)",
                        ft=self.factor[0],
                        fh=self.factor[1],
                        fw=self.factor[2],
                    )
                    residual = residual[:, :, 1:]  # remove 1st frame TODO: this may not be wise
        elif self.factor[0] == 1:
            x = x.repeat_interleave(self.factor[1] * self.factor[2], dim=1)
            residual = rearrange(x, "B (C fh fw) T H W -> B C T (H fh) (W fw)", fh=self.factor[1], fw=self.factor[2])
        else:
            raise NotImplementedError

        return residual


class UpsampleCausal3D(nn.Module):
    """
    A 3D upsampling layer with an optional convolution.
    """

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        bias=True,
        upsample_factor=(2, 2, 2),
        add_residual=False,
        slice_t=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.upsample_factor = upsample_factor
        self.conv = CausalConv3d(self.channels, self.out_channels, kernel_size=kernel_size, bias=bias)
        self.add_residual = add_residual
        if self.add_residual:
            self.shortcut = ChannelDuplicatingPixelShuffleUpSampleLayer(factor=upsample_factor, slice_t=slice_t)

    def forward(
        self,
        input_tensor: torch.FloatTensor,
    ) -> torch.FloatTensor:
        assert input_tensor.shape[1] == self.channels

        #######################
        # handle hidden states
        #######################
        hidden_states = input_tensor
        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # dtype = hidden_states.dtype
        # if dtype == torch.bfloat16:
        #     hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # interpolate H & W only for the first frame; interpolate T & H & W for the rest
        T = hidden_states.size(2)
        first_h, other_h = hidden_states.split((1, T - 1), dim=2)
        # process non-1st frames
        if T > 1:
            other_h = chunk_nearest_interpolate(other_h, scale_factor=self.upsample_factor)
        # proess 1st fram
        first_h = first_h.squeeze(2)
        first_h = chunk_nearest_interpolate(first_h, scale_factor=self.upsample_factor[1:])
        first_h = first_h.unsqueeze(2)
        # concat together
        if T > 1:
            hidden_states = torch.cat((first_h, other_h), dim=2)
        else:
            hidden_states = first_h

        # If the input is bfloat16, we cast back to bfloat16
        # if dtype == torch.bfloat16:
        #     hidden_states = hidden_states.to(dtype)

        hidden_states = self.conv(hidden_states)

        #######################
        # handle residual
        #######################
        if self.add_residual:
            residual = self.shortcut(input_tensor)
            hidden_states += residual

        return hidden_states


class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    """
    residual for downsample layer;
    if has downsample in T dim, add reshaping for T as well.
    Note: (T-1),H,W must be multiples of 2
    """

    def __init__(
        self,
        factor=(1, 2, 2),  # can be (1,2,2) or (2,2,2)
        slice_t=False,  # either slice T or pad T if need to reduce the T dimension
    ):
        super().__init__()
        self.factor = factor
        self.slice_t = slice_t
        self.time_causal_padding = (0, 0, 0, 0, 1, 0)  # W, H, T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.factor[0] == 1 or self.factor[0] == 2, f"unsupported temporal reduction {self.factor[0]}"
        # shape check
        T, H, W = x.shape[-3:]
        assert (
            (T - 1) % self.factor[0] == H % self.factor[1] == W % self.factor[2] == 0
        ), f"{T}-1, {W}, {H} not divisible by {self.factor}"
        if self.factor[0] == 2:  # temporal reduction
            if self.slice_t:
                if T > 1:  # video
                    # slice T and process differently
                    first_f, other_f = x.split((1, T - 1), dim=2)
                    first_f = rearrange(
                        first_f, "B C T (H fh) (W fw) -> B C (fh fw) T H W", fw=self.factor[1], fh=self.factor[2]
                    )
                    first_f = first_f.mean(dim=2)
                    other_f = rearrange(
                        other_f,
                        "B C (T ft) (H fh) (W fw) -> B C (ft fh fw) T H W",
                        ft=self.factor[0],
                        fw=self.factor[1],
                        fh=self.factor[2],
                    )
                    other_f = other_f.mean(dim=2)
                    residual = torch.cat((first_f, other_f), dim=2)
                else:  # image, only work on H & W
                    x = rearrange(x, "B C T (H fh) (W fw) -> B C (fh fw) T H W", fw=self.factor[1], fh=self.factor[2])
                    residual = x.mean(dim=2)
            else:  # use padding to handle temporal reduction
                x = F.pad(x, self.time_causal_padding, mode="replicate")
                # reshape and take average for shortcut
                x = rearrange(
                    x,
                    "B C (T ft) (H fh) (W fw) -> B C (ft fh fw) T H W",
                    ft=self.factor[0],
                    fw=self.factor[1],
                    fh=self.factor[2],
                )
                residual = x.mean(dim=2)
        elif self.factor[0] == 1:  # no temporal reduction
            # reshape and take average for shortcut
            x = rearrange(x, "B C T (H fh) (W fw) -> B C (fh fw) T H W", fw=self.factor[1], fh=self.factor[2])
            residual = x.mean(dim=2)
        else:
            raise NotImplementedError

        return residual


class DownsampleCausal3D(nn.Module):
    """
    A 3D downsampling layer with an optional convolution.
    """

    def __init__(
        self,
        channels: int,
        kernel_size=3,
        bias=True,
        stride=2,
        add_residual=False,
        slice_t=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = channels
        self.conv = CausalConv3d(self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
        self.add_residual = add_residual
        if self.add_residual:
            self.shortcut = PixelUnshuffleChannelAveragingDownSampleLayer(factor=stride, slice_t=slice_t)

    def forward(self, input_tensor: torch.FloatTensor) -> torch.FloatTensor:
        assert input_tensor.shape[1] == self.channels
        hidden_states = self.conv(input_tensor)

        if self.add_residual:
            residual = self.shortcut(input_tensor)
            hidden_states += residual

        return hidden_states


class ResnetBlockCausal3D(nn.Module):
    r"""
    A Resnet block.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        conv_shortcut_bias: bool = True,
        conv_3d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, stride=1)
        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_3d_out_channels = conv_3d_out_channels or out_channels
        self.conv2 = CausalConv3d(out_channels, conv_3d_out_channels, kernel_size=3, stride=1)

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None

        self.use_in_shortcut = self.in_channels != conv_3d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = CausalConv3d(
                in_channels,
                conv_3d_out_channels,
                kernel_size=1,
                stride=1,
                bias=conv_shortcut_bias,
            )

    def forward(
        self,
        input_tensor: torch.FloatTensor,
    ) -> torch.FloatTensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class UNetMidBlockCausal3D(nn.Module):
    """
    A 3D UNet mid-block [`UNetMidBlockCausal3D`] with multiple residual blocks and optional attention blocks.
    """

    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: Optional[int] = None,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        if attn_groups is None:
            attn_groups = resnet_groups

        # there is always at least one resnet
        resnets = [
            ResnetBlockCausal3D(
                in_channels=in_channels,
                out_channels=in_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.FloatTensor, attention_mask: Optional[torch.Tensor]) -> torch.FloatTensor:
        hidden_states = self.resnets[0](hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                B, C, T, H, W = hidden_states.shape
                hidden_states = rearrange(hidden_states, "b c f h w -> b (f h w) c")
                hidden_states = attn(hidden_states, attention_mask=attention_mask)
                hidden_states = rearrange(hidden_states, "b (f h w) c -> b c f h w", f=T, h=H, w=W)
            hidden_states = resnet(hidden_states)

        return hidden_states


class DownEncoderBlockCausal3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_stride: int = 2,
        add_residual: bool = False,
        slice_t: bool = False,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    DownsampleCausal3D(
                        out_channels,
                        stride=downsample_stride,
                        add_residual=add_residual,
                        slice_t=slice_t,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = auto_grad_checkpoint(resnet, hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = auto_grad_checkpoint(downsampler, hidden_states)

        return hidden_states


class UpDecoderBlockCausal3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        upsample_scale_factor=(2, 2, 2),
        add_residual: bool = False,
        slice_t: bool = False,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlockCausal3D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    UpsampleCausal3D(
                        out_channels,
                        out_channels=out_channels,
                        upsample_factor=upsample_scale_factor,
                        add_residual=add_residual,
                        slice_t=slice_t,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.resolution_idx = resolution_idx

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = auto_grad_checkpoint(resnet, hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = auto_grad_checkpoint(upsampler, hidden_states)

        return hidden_states
