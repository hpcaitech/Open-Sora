from typing import List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from colossalai.device.device_mesh import DeviceMesh
from colossalai.shardformer.layer._operation import (
    gather_forward_split_backward,
    reduce_forward,
    split_forward_gather_backward,
)
from colossalai.shardformer.layer.parallel_module import ParallelModule
from colossalai.tensor.d_tensor.api import (
    distribute_tensor,
    is_distributed_tensor,
    shard_rowwise,
    sharded_tensor_to_existing_param,
)
from colossalai.tensor.d_tensor.sharding_spec import ShardingSpec
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter

from .utils import ChannelChunkConv3d, channel_chunk_conv3d


def shard_channelwise(
    tensor: torch.Tensor, group_or_device_mesh: Union[ProcessGroup, DeviceMesh] = None
) -> torch.Tensor:
    """
    Shard the second dim of the given tensor.

    Args:
        tensor (torch.Tensor): The tensor to be sharded.
        group_or_device_mesh (Union[ProcessGroup, DeviceMesh], optional): The group or device mesh to shard the tensor.
            If None, the tensor will be sharded with respect to the global process group.
            Defaults to None.
        inplace (bool, optional): Whether to shard the tensor in-place. Defaults to False.

    Returns:
        torch.Tensor: The sharded tensor.
    """
    # if the group_or_device_mesh is None, we shard the tensor with respect to the global process group
    if group_or_device_mesh is None:
        group_or_device_mesh = dist.GroupMember.WORLD

    if isinstance(group_or_device_mesh, ProcessGroup):
        device_mesh = DeviceMesh.from_process_group(group_or_device_mesh)
    else:
        assert len(group_or_device_mesh.shape) == 1, "Only 1D DeviceMesh is accepted for row-wise sharding."
        device_mesh = group_or_device_mesh
    sharding_spec = ShardingSpec(dim_size=tensor.dim(), dim_partition_dict={1: [0]})

    return distribute_tensor(tensor, device_mesh, sharding_spec)


class Conv3dTPCol(nn.Conv3d):
    """Conv3d with column-wise tensor parallelism. This is only for inference."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        tp_group=None,
        gather_output: bool = False,
        weight: Optional[Parameter] = None,
        bias_: Optional[Parameter] = None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )
        self.tp_group = tp_group
        self.gather_output = gather_output
        self.tp_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)

        # sanity check
        if weight is not None:
            assert not bias or bias_ is not None, "bias_ must be provided if bias is True when weight is not None"
        else:
            assert bias_ is None, "bias_ must be None if weight is None"

        # Parameters.
        if weight is None:
            assert weight is not None, "weight must be provided"
        else:
            weight.data = weight.data.to(device=device, dtype=dtype)
            self.weight = weight

        if not is_distributed_tensor(self.weight):
            sharded_weight = shard_rowwise(self.weight.data, self.tp_group)
            sharded_tensor_to_existing_param(sharded_weight, self.weight)

        if bias:
            if bias_ is None:
                assert bias is not None, "bias must be provided"
            else:
                bias_.data = bias_.data.to(device=device, dtype=dtype)
                self.bias = bias_
            if not is_distributed_tensor(self.bias):
                sharded_bias = shard_rowwise(self.bias.data, self.tp_group)
                sharded_tensor_to_existing_param(sharded_bias, self.bias)
        else:
            self.bias = None

    @staticmethod
    def from_native_module(
        module: nn.Conv3d, process_group: Union[ProcessGroup, List[ProcessGroup]], **kwargs
    ) -> ParallelModule:
        r"""
        Convert a native PyTorch conv3d layer to a tensor parallelized layer.
        """

        # ensure only one process group is passed
        if isinstance(process_group, (list, tuple)):
            assert len(process_group) == 1, f"Expected only one process group, got {len(process_group)}."
            process_group = process_group[0]

        conv3d_tp = Conv3dTPCol(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            device=module.weight.device,
            dtype=module.weight.dtype,
            tp_group=process_group,
            weight=module.weight,
            bias_=module.bias,
            **kwargs,
        )
        return conv3d_tp

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = None
        if self.bias is not None:
            bias = self.bias
        out = channel_chunk_conv3d(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            ChannelChunkConv3d.CONV3D_NUMEL_LIMIT,
        )
        if not self.gather_output:
            return out
        gathered_out = gather_forward_split_backward(out, 1, self.tp_group)
        return gathered_out


class Conv3dTPRow(nn.Conv3d):
    """Conv3d with row-wise tensor parallelism. This is only for inference."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        tp_group=None,
        split_input: bool = False,
        split_output: bool = False,
        weight: Optional[Parameter] = None,
        bias_: Optional[Parameter] = None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )
        self.tp_group = tp_group
        self.split_input = split_input
        self.split_output = split_output
        self.tp_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)

        # sanity check
        if weight is not None:
            assert not bias or bias_ is not None, "bias_ must be provided if bias is True when weight is not None"
        else:
            assert bias_ is None, "bias_ must be None if weight is None"

        # Parameters.
        if weight is None:
            assert weight is not None, "weight must be provided"
        else:
            weight.data = weight.data.to(device=device, dtype=dtype)
            self.weight = weight

        if not is_distributed_tensor(self.weight):
            sharded_weight = shard_channelwise(self.weight.data, self.tp_group)
            sharded_tensor_to_existing_param(sharded_weight, self.weight)

        if bias:
            if bias_ is None:
                assert bias is not None, "bias must be provided"
            else:
                bias_.data = bias_.data.to(device=device, dtype=dtype)
                self.bias = bias_
        else:
            self.bias = None

    @staticmethod
    def from_native_module(
        module: nn.Conv3d, process_group: Union[ProcessGroup, List[ProcessGroup]], **kwargs
    ) -> ParallelModule:
        r"""
        Convert a native PyTorch conv3d layer to a tensor parallelized layer.
        """

        conv3d_tp = Conv3dTPRow(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            device=module.weight.device,
            dtype=module.weight.dtype,
            tp_group=process_group,
            weight=module.weight,
            bias_=module.bias,
            **kwargs,
        )

        return conv3d_tp

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.split_input:
            input = split_forward_gather_backward(input, 1, self.tp_group)
        weight = self.weight
        out = channel_chunk_conv3d(
            input,
            weight,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            ChannelChunkConv3d.CONV3D_NUMEL_LIMIT,
        )
        # del input
        out = reduce_forward(out, self.tp_group)
        if self.bias is not None:
            out = out + self.bias[:, None, None, None]
        if self.split_output:
            out = split_forward_gather_backward(out, 1, self.tp_group)
        return out


class Conv2dTPRow(nn.Conv2d):
    """Conv2d with row-wise tensor parallelism. This is only for inference."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        tp_group=None,
        split_input: bool = False,
        split_output: bool = False,
        weight: Optional[Parameter] = None,
        bias_: Optional[Parameter] = None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )
        self.tp_group = tp_group
        self.split_input = split_input
        self.split_output = split_output
        self.tp_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)

        # sanity check
        if weight is not None:
            assert not bias or bias_ is not None, "bias_ must be provided if bias is True when weight is not None"
        else:
            assert bias_ is None, "bias_ must be None if weight is None"

        # Parameters.
        if weight is None:
            assert weight is not None, "weight must be provided"
        else:
            weight.data = weight.data.to(device=device, dtype=dtype)
            self.weight = weight

        if not is_distributed_tensor(self.weight):
            sharded_weight = shard_channelwise(self.weight.data, self.tp_group)
            sharded_tensor_to_existing_param(sharded_weight, self.weight)

        if bias:
            if bias_ is None:
                assert bias is not None, "bias must be provided"
            else:
                bias_.data = bias_.data.to(device=device, dtype=dtype)
                self.bias = bias_
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.split_input:
            input = split_forward_gather_backward(input, 1, self.tp_group)
        weight = self.weight
        out = F.conv2d(
            input,
            weight,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        # del input
        dist.all_reduce(out, group=self.tp_group)
        if self.bias is not None:
            out += self.bias[:, None, None]
        if self.split_output:
            out = split_forward_gather_backward(out, 1, self.tp_group)
        return out

    @staticmethod
    def from_native_module(
        module: nn.Conv2d, process_group: Union[ProcessGroup, List[ProcessGroup]], **kwargs
    ) -> ParallelModule:
        r"""
        Convert a native PyTorch conv2d layer to a tensor parallelized layer.
        """

        conv2d_tp = Conv2dTPRow(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            device=module.weight.device,
            dtype=module.weight.dtype,
            tp_group=process_group,
            weight=module.weight,
            bias_=module.bias,
            **kwargs,
        )
        conv2d_tp.weight = module.weight
        conv2d_tp.bias = module.bias
        return conv2d_tp


class Conv1dTPRow(nn.Conv1d):
    """Conv1d with row-wise tensor parallelism. This is only for inference."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        tp_group=None,
        split_input: bool = False,
        split_output: bool = False,
        weight: Optional[Parameter] = None,
        bias_: Optional[Parameter] = None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )
        self.tp_group = tp_group
        self.split_input = split_input
        self.split_output = split_output
        self.tp_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)

        # sanity check
        if weight is not None:
            assert not bias or bias_ is not None, "bias_ must be provided if bias is True when weight is not None"
        else:
            assert bias_ is None, "bias_ must be None if weight is None"

        # Parameters.
        if weight is None:
            assert weight is not None, "weight must be provided"
        else:
            weight.data = weight.data.to(device=device, dtype=dtype)
            self.weight = weight

        if not is_distributed_tensor(self.weight):
            sharded_weight = shard_channelwise(self.weight.data, self.tp_group)
            sharded_tensor_to_existing_param(sharded_weight, self.weight)

        if bias:
            if bias_ is None:
                assert bias is not None, "bias must be provided"
            else:
                bias_.data = bias_.data.to(device=device, dtype=dtype)
                self.bias = bias_
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.split_input:
            input = split_forward_gather_backward(input, 1, self.tp_group)

        weight = self.weight
        out = F.conv1d(
            input,
            weight,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        # del input
        dist.all_reduce(out, group=self.tp_group)
        if self.bias is not None:
            out += self.bias[:, None]
        if self.split_output:
            out = split_forward_gather_backward(out, 1, self.tp_group)
        return out

    @staticmethod
    def from_native_module(
        module: nn.Conv1d, process_group: Union[ProcessGroup, List[ProcessGroup]], **kwargs
    ) -> ParallelModule:
        r"""
        Convert a native PyTorch conv1d layer to a tensor parallelized layer.
        """

        conv1d_tp = Conv1dTPRow(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            device=module.weight.device,
            dtype=module.weight.dtype,
            tp_group=process_group,
            weight=module.weight,
            bias_=module.bias,
            **kwargs,
        )
        conv1d_tp.weight = module.weight
        conv1d_tp.bias = module.bias
        return conv1d_tp


class GroupNormTP(nn.GroupNorm):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 0.00001,
        affine: bool = True,
        device=None,
        dtype=None,
        tp_group=None,
        weight: Optional[Parameter] = None,
        bias: Optional[Parameter] = None,
    ) -> None:
        super().__init__(num_groups, num_channels, eps, affine, device, dtype)
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)

        if affine:
            assert weight is not None, "weight must be provided"
            weight.data = weight.data.to(device=device, dtype=dtype)
            self.weight = weight
            if not is_distributed_tensor(self.weight):
                sharded_weight = shard_rowwise(self.weight.data, self.tp_group)
                sharded_tensor_to_existing_param(sharded_weight, self.weight)

            assert bias is not None, "bias must be provided"
            bias.data = bias.data.to(device=device, dtype=dtype)
            self.bias = bias
            if not is_distributed_tensor(self.bias):
                sharded_bias = shard_rowwise(self.bias.data, self.tp_group)
                sharded_tensor_to_existing_param(sharded_bias, self.bias)
        else:
            self.weight = None
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.group_norm(
            input,
            self.num_groups // self.tp_size,
            self.weight,
            self.bias,
            self.eps,
        )

    @staticmethod
    def from_native_module(
        module: nn.GroupNorm, process_group: Union[ProcessGroup, List[ProcessGroup]], **kwargs
    ) -> ParallelModule:
        r"""
        Convert a native PyTorch nn.GroupNorm layer to a tensor parallelized layer.
        """

        group_norm_tp = GroupNormTP(
            num_groups=module.num_groups,
            num_channels=module.num_channels,
            eps=module.eps,
            affine=module.affine,
            device=module.weight.device,
            dtype=module.weight.dtype,
            tp_group=process_group,
            weight=module.weight,
            bias=module.bias,
            **kwargs,
        )
        return group_norm_tp
