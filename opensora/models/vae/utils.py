import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

NUMEL_LIMIT = 2**30


def ceil_to_divisible(n: int, dividend: int) -> int:
    return math.ceil(dividend / (dividend // n))


def chunked_avg_pool1d(
    input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True
):
    n_chunks = math.ceil(input.numel() / NUMEL_LIMIT)
    if n_chunks == 1:
        return F.avg_pool1d(
            input, kernel_size, stride, padding, ceil_mode, count_include_pad
        )
    else:
        l_in = input.shape[-1]
        l_out = math.floor((l_in + 2 * padding - kernel_size) / stride + 1)
        output_shape = list(input.shape)
        output_shape[-1] = l_out
        out_list = []

        for inp_chunk in input.chunk(n_chunks, dim=0):
            out_chunk = F.avg_pool1d(
                inp_chunk, kernel_size, stride, padding, ceil_mode, count_include_pad
            )
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
    input_shape: torch.Size,
    out_channels: int,
    kernel_size: list,
    stride: list,
    padding: int,
    dilation: list,
) -> list:
    output_shape = [out_channels]
    if len(input_shape) == 5:
        output_shape.insert(0, input_shape[0])
    if padding == "same":
        return output_shape + list(input_shape[-3:])
    if padding == "valid":
        padding = (0, 0, 0)
    for i, d in enumerate(input_shape[-3:]):
        d_out = math.floor(
            (d + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]
            + 1
        )
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
    out_channels = weight.shape[0]
    kernel_size = weight.shape[2:]
    output_shape = get_conv3d_output_shape(
        input.shape, out_channels, kernel_size, stride, padding, dilation
    )
    output_numel = math.prod(output_shape)
    if input.numel() <= numel_limit and output_numel <= numel_limit:
        return F.conv3d(input, weight, bias, stride, padding, dilation, groups)

    channel_dim = 1 if input.dim() == 5 else 0
    input_channels = input.shape[channel_dim]

    if groups > 1:
        in_channels_per_group = input_channels // groups
        out_channels_per_group = out_channels // groups
        input_numel_per_group = input.numel() // groups
        output_numel_per_group = output_numel // groups
        groups_per_chunk = max(
            1,
            min(
                groups,
                numel_limit // input_numel_per_group,
                numel_limit // output_numel_per_group,
            ),
        )

        outputs = []
        for group_start in range(0, groups, groups_per_chunk):
            chunk_groups = min(groups_per_chunk, groups - group_start)
            input_start = group_start * in_channels_per_group
            output_start = group_start * out_channels_per_group
            input_chunk = input.narrow(
                channel_dim, input_start, chunk_groups * in_channels_per_group
            )
            weight_chunk = weight.narrow(
                0, output_start, chunk_groups * out_channels_per_group
            )
            bias_chunk = (
                bias.narrow(0, output_start, chunk_groups * out_channels_per_group)
                if bias is not None
                else None
            )
            outputs.append(
                channel_chunk_conv3d(
                    input_chunk,
                    weight_chunk,
                    bias_chunk,
                    stride,
                    padding,
                    dilation,
                    chunk_groups,
                    numel_limit,
                )
            )
        return torch.cat(outputs, dim=channel_dim)

    input_numel_per_channel = input.numel() // input_channels
    output_numel_per_channel = output_numel // out_channels
    in_channels_per_chunk = max(
        1, min(input_channels, numel_limit // input_numel_per_channel)
    )
    out_channels_per_chunk = max(
        1, min(out_channels, numel_limit // output_numel_per_channel)
    )
    outputs = []
    for output_start in range(0, out_channels, out_channels_per_chunk):
        chunk_out_channels = min(out_channels_per_chunk, out_channels - output_start)
        weight_chunk = weight.narrow(0, output_start, chunk_out_channels)
        output = None
        output_dtype = None
        for input_start in range(0, input_channels, in_channels_per_chunk):
            chunk_in_channels = min(in_channels_per_chunk, input_channels - input_start)
            input_chunk = input.narrow(channel_dim, input_start, chunk_in_channels)
            weight_shard = weight_chunk.narrow(1, input_start, chunk_in_channels)
            partial = F.conv3d(
                input_chunk, weight_shard, None, stride, padding, dilation, 1
            )
            if output_dtype is None:
                output_dtype = partial.dtype
                accumulation_dtype = (
                    torch.float32
                    if output_dtype in (torch.float16, torch.bfloat16)
                    else output_dtype
                )
            partial = partial.to(accumulation_dtype)
            output = partial if output is None else output + partial

        if bias is not None:
            bias_chunk = bias.narrow(0, output_start, chunk_out_channels).to(
                accumulation_dtype
            )
            bias_shape = [1] * output.dim()
            bias_shape[channel_dim] = chunk_out_channels
            output = output + bias_chunk.view(bias_shape)
        outputs.append(output.to(output_dtype))
    return torch.cat(outputs, dim=channel_dim)


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
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device, dtype=self.mean.dtype
            )

    def sample(self):
        # torch.randn: standard normal distribution
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device, dtype=self.mean.dtype
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:  # SCH: assumes other is a standard normal distribution
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 3, 4],
                ).flatten(0)
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

    def forward(self, input: Tensor) -> Tensor:
        padding = self.padding
        if self.padding_mode != "zeros":
            input = F.pad(
                input, self._reversed_padding_repeated_twice, mode=self.padding_mode
            )
            padding = (0, 0, 0)
        return channel_chunk_conv3d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding,
            self.dilation,
            self.groups,
            self.CONV3D_NUMEL_LIMIT,
        )


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def pad_for_conv3d(
    x: torch.Tensor, width_pad: int, height_pad: int, time_pad: int
) -> torch.Tensor:
    if width_pad > 0 or height_pad > 0:
        x = F.pad(
            x, (width_pad, width_pad, height_pad, height_pad), mode="constant", value=0
        )
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
        assert (
            time_kernel_size == height_kernel_size == width_kernel_size
        ), "only support cubic kernel size"
        if time_kernel_size == 3:
            self.pad = pad_for_conv3d_kernel_3x3x3
        else:
            assert (
                time_kernel_size == 1
            ), f"only support kernel size 1/3 for now, got {kernel_size}"
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
        self.conv = ChannelChunkConv3d(
            in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
        )
