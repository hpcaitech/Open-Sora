from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opensora.acceleration.checkpoint import auto_grad_checkpoint, checkpoint
from opensora.models.hunyuan_vae.unet_causal_3d_blocks import (
    CausalConv3d,
    ResnetBlockCausal3D,
    UNetMidBlockCausal3D,
    chunk_nearest_interpolate,
)


def pixel_shuffle_channel_averaging(
    input,
    channel_factor=2,
    factor=(1, 2, 2),
):
    B, C, T, H, W = input.size()
    assert T % factor[0] == 0 and H % factor[1] == 0 and W % factor[2] == 0
    assert (factor[0] * factor[1] * factor[2]) % channel_factor == 0
    output = input.view(B, C, T // factor[0], factor[0], H // factor[1], factor[1], W // factor[2], factor[2])
    output = output.permute(0, 1, 3, 5, 7, 2, 4, 6)
    output = output.contiguous().view(B, C * channel_factor, -1, T // factor[0], H // factor[1], W // factor[2])
    output = output.mean(dim=2)
    return output


def channel_repeat_pixel_shuffle(
    input,
    channel_factor=2,
    factor=(1, 2, 2),
):
    B, C, T, H, W = input.size()
    assert factor[0] * factor[1] * factor[2] % channel_factor == 0
    repeat = factor[0] * factor[1] * factor[2] // channel_factor
    output = input.repeat_interleave(repeat, dim=1)
    output = output.view(B, C // channel_factor, factor[0], factor[1], factor[2], T, H, W)
    output = output.permute(0, 1, 5, 2, 6, 3, 7, 4)
    output = output.contiguous().view(B, C // channel_factor, T * factor[0], H * factor[1], W * factor[2])
    return output


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
        channel_factor=2,
    ):
        super().__init__()
        self.factor = factor
        self.slice_t = slice_t
        self.time_causal_padding = (0, 0, 0, 0, 1, 0)  # W, H, T
        self.channel_factor = channel_factor

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
                    first_f = pixel_shuffle_channel_averaging(
                        first_f, channel_factor=self.channel_factor, factor=(1, self.factor[1], self.factor[2])
                    )
                    other_f = pixel_shuffle_channel_averaging(
                        other_f, channel_factor=self.channel_factor, factor=self.factor
                    )
                    residual = torch.cat((first_f, other_f), dim=2)
                else:  # image, only work on H & W
                    residual = pixel_shuffle_channel_averaging(
                        x, channel_factor=self.channel_factor, factor=self.factor
                    )
            else:  # use padding to handle temporal reduction
                x = F.pad(x, self.time_causal_padding, mode="replicate")
                # reshape and take average for shortcut
                residual = pixel_shuffle_channel_averaging(x, channel_factor=self.channel_factor, factor=self.factor)
        elif self.factor[0] == 1:  # no temporal reduction
            # reshape and take average for shortcut
            residual = pixel_shuffle_channel_averaging(x, channel_factor=self.channel_factor, factor=self.factor)
        else:
            raise NotImplementedError

        return residual


class ChannelDuplicatingPixelShuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        factor=(1, 2, 2),
        slice_t=False,  # either slice T or pad T
        channel_factor=2,
    ):
        super().__init__()
        self.factor = factor
        self.slice_t = slice_t
        self.channel_factor = channel_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(2)
        if self.factor[0] == 2:
            if T == 1:  # image
                residual = channel_repeat_pixel_shuffle(
                    x, channel_factor=self.channel_factor, factor=(1, self.factor[1], self.factor[2])
                )
            else:  # video
                if self.slice_t:
                    # slice T and process differently
                    first_f, other_f = x.split((1, T - 1), dim=2)
                    first_f = channel_repeat_pixel_shuffle(
                        first_f, channel_factor=self.channel_factor, factor=(1, self.factor[1], self.factor[2])
                    )
                    other_f = channel_repeat_pixel_shuffle(
                        other_f, channel_factor=self.channel_factor, factor=self.factor
                    )
                    residual = torch.cat((first_f, other_f), dim=2)
                else:
                    residual = channel_repeat_pixel_shuffle(x, channel_factor=self.channel_factor, factor=self.factor)
                    residual = residual[:, :, 1:]  # remove 1st frame TODO: this may not be wise
        elif self.factor[0] == 1:
            residual = channel_repeat_pixel_shuffle(x, channel_factor=self.channel_factor, factor=self.factor)
        else:
            raise NotImplementedError

        return residual


class DownsampleCausal3D(nn.Module):
    """
    A 3D downsampling layer with an optional convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        bias=True,
        stride=2,
        add_residual=False,
        slice_t=False,
    ):
        super().__init__()
        self.channels = in_channels
        self.out_channels = out_channels
        assert self.out_channels % self.channels == 0
        self.conv = CausalConv3d(self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
        self.add_residual = add_residual
        if self.add_residual:
            self.shortcut = PixelUnshuffleChannelAveragingDownSampleLayer(
                factor=stride, slice_t=slice_t, channel_factor=self.out_channels // self.channels
            )

    def forward(self, input_tensor: torch.FloatTensor) -> torch.FloatTensor:
        assert input_tensor.shape[1] == self.channels
        hidden_states = self.conv(input_tensor)

        if self.add_residual:
            residual = self.shortcut(input_tensor)
            hidden_states += residual

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
            if add_downsample:
                target_channel = in_channels
            else:
                target_channel = in_channels if i < num_layers - 1 else out_channels
            resnets.append(
                ResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=target_channel,
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
                        in_channels,
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


class ChannelEncoderCausal3D(nn.Module):
    r"""
    The `EncoderCausal3D` layer of a variational autoencoder that encodes its input into a latent representation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
        time_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
        dropout: float = 0.0,
        add_residual: bool = False,
        slice_t: bool = False,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = CausalConv3d(in_channels, block_out_channels[0], kernel_size=3, stride=1)
        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, _ in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_downsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_downsample_layers = int(np.log2(time_compression_ratio))

            if time_compression_ratio == 1:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)
                add_time_downsample = False
            elif time_compression_ratio == 4:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)
                add_time_downsample = bool(
                    i >= (len(block_out_channels) - 1 - num_time_downsample_layers) and not is_final_block
                )
            elif time_compression_ratio == 8:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)
                add_time_downsample = bool(i < num_spatial_downsample_layers)
            else:
                raise ValueError(f"Unsupported time_compression_ratio: {time_compression_ratio}.")

            downsample_stride_HW = (2, 2) if add_spatial_downsample else (1, 1)
            downsample_stride_T = (2,) if add_time_downsample else (1,)
            downsample_stride = tuple(downsample_stride_T + downsample_stride_HW)
            down_block = DownEncoderBlockCausal3D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                dropout=dropout,
                add_downsample=bool(add_spatial_downsample or add_time_downsample),
                downsample_stride=downsample_stride,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                add_residual=add_residual,
                slice_t=slice_t,
            )

            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlockCausal3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = CausalConv3d(block_out_channels[-1], conv_out_channels, kernel_size=3)

    def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        r"""The forward method of the `EncoderCausal3D` class."""
        assert len(sample.shape) == 5, "The input tensor should have 5 dimensions"

        sample = self.conv_in(sample)

        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)

        # middle
        sample = auto_grad_checkpoint(self.mid_block, sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class UpsampleCausal3D(nn.Module):
    """
    A 3D upsampling layer with an optional convolution.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias=True,
        upsample_factor=(2, 2, 2),
        add_residual=False,
        slice_t=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        assert channels % out_channels == 0
        self.upsample_factor = upsample_factor
        self.conv = CausalConv3d(self.channels, self.out_channels, kernel_size=kernel_size, bias=bias)
        self.add_residual = add_residual
        if self.add_residual:
            self.shortcut = ChannelDuplicatingPixelShuffleUpSampleLayer(
                factor=upsample_factor, slice_t=slice_t, channel_factor=channels // out_channels
            )

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
            if add_upsample:
                target_channel = in_channels
            else:
                target_channel = in_channels if i < num_layers - 1 else out_channels
            resnets.append(
                ResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=target_channel,
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
                        in_channels,
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


class ChannelDecoderCausal3D(nn.Module):
    r"""
    The `DecoderCausal3D` layer of a variational autoencoder that decodes its latent representation into an output sample.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention=True,
        time_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
        dropout: float = 0.0,
        add_residual: bool = False,
        slice_t: bool = False,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = CausalConv3d(in_channels, block_out_channels[-1], kernel_size=3, stride=1)
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = UNetMidBlockCausal3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, _ in enumerate(block_out_channels):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_upsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_upsample_layers = int(np.log2(time_compression_ratio))

            if time_compression_ratio == 1:
                add_spatial_upsample = bool(i < num_spatial_upsample_layers)
                add_time_upsample = False
            elif time_compression_ratio == 4:
                add_spatial_upsample = bool(i < num_spatial_upsample_layers)
                add_time_upsample = bool(
                    i >= len(block_out_channels) - 1 - num_time_upsample_layers and not is_final_block
                )
            elif time_compression_ratio == 8:
                add_spatial_upsample = bool(i < num_spatial_upsample_layers)
                add_time_upsample = bool(i < num_spatial_upsample_layers)
            else:
                raise ValueError(f"Unsupported time_compression_ratio: {time_compression_ratio}.")

            upsample_scale_factor_HW = (2, 2) if add_spatial_upsample else (1, 1)
            upsample_scale_factor_T = (2,) if add_time_upsample else (1,)
            upsample_scale_factor = tuple(upsample_scale_factor_T + upsample_scale_factor_HW)
            up_block = UpDecoderBlockCausal3D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                resolution_idx=None,
                dropout=dropout,
                add_upsample=bool(add_spatial_upsample or add_time_upsample),
                upsample_scale_factor=upsample_scale_factor,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                add_residual=add_residual,
                slice_t=slice_t,
            )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(block_out_channels[0], out_channels, kernel_size=3)

    def post_process(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        return sample

    def forward(
        self,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        r"""The forward method of the `DecoderCausal3D` class."""
        assert len(sample.shape) == 5, "The input tensor should have 5 dimensions."

        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

        # middle
        sample = auto_grad_checkpoint(self.mid_block, sample)
        sample = sample.to(upscale_dtype)

        # up
        for up_block in self.up_blocks:
            sample = up_block(sample)

        # post-process
        if getattr(self, "grad_checkpointing", False):
            sample = checkpoint(self.post_process, sample, use_reentrant=True)
        else:
            sample = self.post_process(sample)

        sample = self.conv_out(sample)

        return sample
