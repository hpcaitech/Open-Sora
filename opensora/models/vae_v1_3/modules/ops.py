# modified from
# https://github.com/bornfly-detachment/asymmetric_magvitv2/blob/main/models/modules/ops.py

import torch


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


def nonlinearity(x, is_training=False):
    if is_training:
        return x * torch.sigmoid(x)
    else:
        x.mul_(torch.sigmoid(x))
        return x


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)
