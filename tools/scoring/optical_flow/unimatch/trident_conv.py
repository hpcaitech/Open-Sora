# Copyright (c) Facebook, Inc. and its affiliates.
# https://github.com/facebookresearch/detectron2/blob/main/projects/TridentNet/tridentnet/trident_conv.py

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair


class MultiScaleTridentConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        strides=1,
        paddings=0,
        dilations=1,
        dilation=1,
        groups=1,
        num_branch=1,
        test_branch_idx=-1,
        bias=False,
        norm=None,
        activation=None,
    ):
        super(MultiScaleTridentConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.num_branch = num_branch
        self.stride = _pair(stride)
        self.groups = groups
        self.with_bias = bias
        self.dilation = dilation
        if isinstance(paddings, int):
            paddings = [paddings] * self.num_branch
        if isinstance(dilations, int):
            dilations = [dilations] * self.num_branch
        if isinstance(strides, int):
            strides = [strides] * self.num_branch
        self.paddings = [_pair(padding) for padding in paddings]
        self.dilations = [_pair(dilation) for dilation in dilations]
        self.strides = [_pair(stride) for stride in strides]
        self.test_branch_idx = test_branch_idx
        self.norm = norm
        self.activation = activation

        assert len({self.num_branch, len(self.paddings), len(self.strides)}) == 1

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, inputs):
        num_branch = self.num_branch if self.training or self.test_branch_idx == -1 else 1
        assert len(inputs) == num_branch

        if self.training or self.test_branch_idx == -1:
            outputs = [
                F.conv2d(input, self.weight, self.bias, stride, padding, self.dilation, self.groups)
                for input, stride, padding in zip(inputs, self.strides, self.paddings)
            ]
        else:
            outputs = [
                F.conv2d(
                    inputs[0],
                    self.weight,
                    self.bias,
                    self.strides[self.test_branch_idx] if self.test_branch_idx == -1 else self.strides[-1],
                    self.paddings[self.test_branch_idx] if self.test_branch_idx == -1 else self.paddings[-1],
                    self.dilation,
                    self.groups,
                )
            ]

        if self.norm is not None:
            outputs = [self.norm(x) for x in outputs]
        if self.activation is not None:
            outputs = [self.activation(x) for x in outputs]
        return outputs
