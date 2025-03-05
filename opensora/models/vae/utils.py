import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

NUMEL_LIMIT = 2**30


def ceil_to_divisible(n: int, dividend: int) -> int:
    return math.ceil(dividend / (dividend // n))


def chunked_avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    n_chunks = math.ceil(input.numel() / NUMEL_LIMIT)
    if n_chunks == 1:
        return F.avg_pool1d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
    else:
        l_in = input.shape[-1]
        l_out = math.floor((l_in + 2 * padding - kernel_size) / stride + 1)
        output_shape = list(input.shape)
        output_shape[-1] = l_out
        out_list = []

        for inp_chunk in input.chunk(n_chunks, dim=0):
            out_chunk = F.avg_pool1d(inp_chunk, kernel_size, stride, padding, ceil_mode, count_include_pad)
            out_list.append(out_chunk)
        return torch.cat(out_list, dim=0)


def chunked_interpolate(input, scale_factor):
    output_shape = list(input.shape)
    output_shape = output_shape[:2] + [int(i * scale_factor) for i in output_shape[2:]]
    n_chunks = math.ceil(torch.Size(output_shape).numel() / NUMEL_LIMIT)
    if n_chunks == 1:
        return F.interpolate(input, scale_factor=scale_factor)
    else:
        out_list = []
        n_chunks += 1
        for inp_chunk in input.chunk(n_chunks, dim=1):
            out_chunk = F.interpolate(inp_chunk, scale_factor=scale_factor)
            out_list.append(out_chunk)
        return torch.cat(out_list, dim=1)


def get_conv3d_output_shape(
    input_shape: torch.Size, out_channels: int, kernel_size: list, stride: list, padding: int, dilation: list
) -> list:
    output_shape = [out_channels]
    if len(input_shape) == 5:
        output_shape.insert(0, input_shape[0])
    for i, d in enumerate(input_shape[-3:]):
        d_out = math.floor((d + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i] + 1)
        output_shape.append(d_out)
    return output_shape


def get_conv3d_n_chunks(numel: int, n_channels: int, numel_limit: int):
    n_chunks = math.ceil(numel / numel_limit)
    n_chunks = ceil_to_divisible(n_chunks, n_channels)
    return n_chunks


def channel_chunk_conv3d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: list,
    padding: list,
    dilation: list,
    groups: int,
    numel_limit: int,
):
    out_channels, in_channels = weight.shape[:2]
    kernel_size = weight.shape[2:]
    output_shape = get_conv3d_output_shape(input.shape, out_channels, kernel_size, stride, padding, dilation)
    n_in_chunks = get_conv3d_n_chunks(input.numel(), in_channels, numel_limit)
    n_out_chunks = get_conv3d_n_chunks(
        np.prod(output_shape),
        out_channels,
        numel_limit,
    )
    if n_in_chunks == 1 and n_out_chunks == 1:
        return F.conv3d(input, weight, bias, stride, padding, dilation, groups)
    # output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
    # outputs = output.chunk(n_out_chunks, dim=1)
    input_shards = input.chunk(n_in_chunks, dim=1)
    weight_chunks = weight.chunk(n_out_chunks)
    output_list = []
    if bias is not None:
        bias_chunks = bias.chunk(n_out_chunks)
    else:
        bias_chunks = [None] * n_out_chunks
    for weight_, bias_ in zip(weight_chunks, bias_chunks):
        weight_shards = weight_.chunk(n_in_chunks, dim=1)
        o = None
        for x, w in zip(input_shards, weight_shards):
            if o is None:
                o = F.conv3d(x, w, None, stride, padding, dilation, groups).float()
            else:
                o += F.conv3d(x, w, None, stride, padding, dilation, groups).float()
        o = o.to(input.dtype)
        if bias_ is not None:
            o += bias_[None, :, None, None, None]
        # inplace operation cannot be used during training
        # output_.copy_(o)
        output_list.append(o)
    return torch.cat(output_list, dim=1)


class DiagonalGaussianDistribution(object):
    def __init__(
        self,
        parameters,
        deterministic=False,
    ):
        """Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device, dtype=self.mean.dtype)

    def sample(self):
        # torch.randn: standard normal distribution
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device, dtype=self.mean.dtype)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:  # SCH: assumes other is a standard normal distribution
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 3, 4]).flatten(0)
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 3, 4],
                ).flatten(0)

    def mode(self):
        return self.mean


class ChannelChunkConv3d(nn.Conv3d):
    CONV3D_NUMEL_LIMIT = 2**31

    def _get_output_numel(self, input_shape: torch.Size) -> int:
        numel = self.out_channels
        if len(input_shape) == 5:
            numel *= input_shape[0]
        for i, d in enumerate(input_shape[-3:]):
            d_out = math.floor(
                (d + 2 * self.padding[i] - self.dilation[i] * (self.kernel_size[i] - 1) - 1) / self.stride[i] + 1
            )
            numel *= d_out
        return numel

    def _get_n_chunks(self, numel: int, n_channels: int):
        n_chunks = math.ceil(numel / ChannelChunkConv3d.CONV3D_NUMEL_LIMIT)
        n_chunks = ceil_to_divisible(n_chunks, n_channels)
        return n_chunks

    def forward(self, input: Tensor) -> Tensor:
        if input.numel() // input.size(0) < ChannelChunkConv3d.CONV3D_NUMEL_LIMIT:
            return super().forward(input)
        n_in_chunks = self._get_n_chunks(input.numel(), self.in_channels)
        n_out_chunks = self._get_n_chunks(self._get_output_numel(input.shape), self.out_channels)
        if n_in_chunks == 1 and n_out_chunks == 1:
            return super().forward(input)
        outputs = []
        input_shards = input.chunk(n_in_chunks, dim=1)
        for weight, bias in zip(self.weight.chunk(n_out_chunks), self.bias.chunk(n_out_chunks)):
            weight_shards = weight.chunk(n_in_chunks, dim=1)
            o = None
            for x, w in zip(input_shards, weight_shards):
                if o is None:
                    o = F.conv3d(x, w, bias, self.stride, self.padding, self.dilation, self.groups)
                else:
                    o += F.conv3d(x, w, None, self.stride, self.padding, self.dilation, self.groups)
            outputs.append(o)
        return torch.cat(outputs, dim=1)


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def pad_for_conv3d(x: torch.Tensor, width_pad: int, height_pad: int, time_pad: int) -> torch.Tensor:
    if width_pad > 0 or height_pad > 0:
        x = F.pad(x, (width_pad, width_pad, height_pad, height_pad), mode="constant", value=0)
    if time_pad > 0:
        x = F.pad(x, (0, 0, 0, 0, time_pad, time_pad), mode="replicate")
    return x


def pad_for_conv3d_kernel_3x3x3(x: torch.Tensor) -> torch.Tensor:
    n_chunks = math.ceil(x.numel() / NUMEL_LIMIT)
    if n_chunks == 1:
        x = F.pad(x, (1, 1, 1, 1), mode="constant", value=0)
        x = F.pad(x, (0, 0, 0, 0, 1, 1), mode="replicate")
    else:
        out_list = []
        n_chunks += 1
        for inp_chunk in x.chunk(n_chunks, dim=1):
            out_chunk = F.pad(inp_chunk, (1, 1, 1, 1), mode="constant", value=0)
            out_chunk = F.pad(out_chunk, (0, 0, 0, 0, 1, 1), mode="replicate")
            out_list.append(out_chunk)
        x = torch.cat(out_list, dim=1)
    return x


class PadConv3D(nn.Module):
    """
    pad the first frame in temporal dimension
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        self.kernel_size = kernel_size

        # == specific padding ==
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        assert time_kernel_size == height_kernel_size == width_kernel_size, "only support cubic kernel size"
        if time_kernel_size == 3:
            self.pad = pad_for_conv3d_kernel_3x3x3
        else:
            assert time_kernel_size == 1, f"only support kernel size 1/3 for now, got {kernel_size}"
            self.pad = lambda x: x

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)
        x = self.conv(x)
        return x


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
class ChannelChunkPadConv3D(PadConv3D):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__(in_channels, out_channels, kernel_size)
        self.conv = ChannelChunkConv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1)
