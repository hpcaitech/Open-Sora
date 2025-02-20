# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# PixArt: https://github.com/PixArt-alpha/PixArt-alpha
# Latte:  https://github.com/Vchitect/Latte
# DiT:    https://github.com/facebookresearch/DiT/tree/main
# GLIDE:  https://github.com/openai/glide-text2im
# MAE:    https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import functools
import math
import warnings
from itertools import chain
from typing import Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import xformers.ops
from einops import rearrange
from timm.models.vision_transformer import Mlp

from opensora.acceleration.communications import all_to_all, split_forward_gather_backward
from opensora.acceleration.parallel_states import get_sequence_parallel_group

approx_gelu = lambda: nn.GELU(approximate="tanh")


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def get_layernorm(hidden_size: torch.Tensor, eps: float, affine: bool, use_kernel: bool):
    if use_kernel:
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(hidden_size, elementwise_affine=affine, eps=eps)
        except ImportError:
            raise RuntimeError("FusedLayerNorm not available. Please install apex.")
    else:
        return nn.LayerNorm(hidden_size, eps, elementwise_affine=affine)


def modulate(norm_func, x, shift, scale):
    # Suppose x is (B, N, D), shift is (B, D), scale is (B, D)
    dtype = x.dtype
    x = norm_func(x.to(torch.float32)).to(dtype)
    x = x * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)
    x = x.to(dtype)
    return x


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def pad_to_multiples(tensor: torch.Tensor, multiples, dims, value=0, extra_pad_on_dims=None):
    """
    Pad the input tensor to the multiples of the given values. Right padding
    """
    assert len(multiples) == len(dims) and len(multiples) > 0 and len(multiples) <= tensor.dim()
    if extra_pad_on_dims is not None:
        assert len(extra_pad_on_dims) == len(multiples) * 2
    n_pads = [0] * tensor.dim() * 2
    for i, (dim, multiple) in enumerate(zip(dims, multiples)):
        n_pads[dim * 2] = (multiple - tensor.size(dim) % multiple) % multiple
        if extra_pad_on_dims is not None:
            n_pads[dim * 2 + 1] += extra_pad_on_dims[i * 2]
            n_pads[dim * 2] += extra_pad_on_dims[i * 2 + 1]
    n_pads = n_pads[::-1]
    output_tensor = F.pad(tensor, n_pads, value=value)
    if output_tensor.shape == tensor.shape:
        padding_mask = None
    else:
        padding_mask = torch.ones_like(tensor, dtype=torch.int)
        padding_mask = F.pad(padding_mask, n_pads, value=0)
    return output_tensor, padding_mask


def remove_padding_nd(tensor: torch.Tensor, origin_sizes, dims):
    """
    Remove padding from the input tensor
    """
    assert len(origin_sizes) == len(dims) and len(origin_sizes) > 0 and len(origin_sizes) <= tensor.dim()
    for size, dim in zip(origin_sizes, dims):
        tensor = tensor.narrow(dim, 0, size)
    return tensor


def split_seq_cat_batch(tensor: torch.Tensor, split_sizes, dims, batch_dim: int = 0) -> torch.Tensor:
    """split tensor in sequence dimension and cat in batch dimension

    Args:
        tensor (torch.Tensor): [B, *N, C]
        split_sizes (_type_): kernel size
        dims (_type_): dim index of "N"
        batch_dim (int, optional): dim of batch. Defaults to 0.

    Returns:
        torch.Tensor: [MB, *K, C], K is kernel size. Total number of dim is the same as input tensor.
    """
    chunks = [tensor]
    for dim, split_size in zip(dims, split_sizes):
        new_chunks = []
        for t in chunks:
            new_chunks.extend(t.split(split_size, dim))
        chunks = new_chunks
    return torch.cat(chunks, batch_dim)


def split_batch_cat_seq(tensor: torch.Tensor, batch_size: int, num_splits, dims, batch_dim: int = 0) -> torch.Tensor:
    """split tensor in batch dimension and cat in sequence dimension

    Args:
        tensor (torch.Tensor): [MB, *K, C]
        batch_size (int): original batch size
        num_splits (_type_): number of splits of each dim
        dims (_type_): dim index of "K"
        batch_dim (int, optional): dim of batch. Defaults to 0.

    Returns:
        torch.Tensor: [B, *N, C]
    """
    chunks = tensor.split(batch_size, batch_dim)
    for dim, num_split in zip(dims[::-1], num_splits[::-1]):
        new_chunks = []
        for i in range(0, len(chunks), num_split):
            group = chunks[i : i + num_split]
            new_chunks.append(torch.cat(group, dim))
        chunks = new_chunks
    assert len(chunks) == 1
    return chunks[0]


# ===============================================
# General-purpose Layers
# ===============================================


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # (B C T H W)
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        rope=None,
        qk_norm_legacy: bool = False,
        kernel_size: Optional[Sequence[int]] = None,
        shift_window: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm_legacy = qk_norm_legacy
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope

        self.is_causal = False

        self.is_causal = False
        self.kernel_size = kernel_size
        self.shift_window = shift_window
        if shift_window and kernel_size is None:
            warnings.warn(f"shift_window is enabled but kernel_size is not set, this may not work as expected")
        if kernel_size is not None and shift_window:
            self.extra_pad_on_dims = tuple(
                chain.from_iterable((k // 2, k - k // 2) if k >= 0 else (0, 0) for k in kernel_size)
            )
        else:
            self.extra_pad_on_dims = None

    def rotate_3d(self, k, ks1, ks2, scale=None):
        k_t, k_h, k_w = k.chunk(3, dim=-1)
        # temporal
        k_t = rearrange(k_t, "b h (k1 k2 T) d -> b h k1 k2 T d", k1=ks1, k2=ks2)
        k_t = self.rotary_emb(k_t)
        k_t = rearrange(k_t, "b h k1 k2 T d -> b h (k1 k2 T) d", k1=ks1, k2=ks2)
        # height
        k_h = rearrange(k_h, "b h (k1 k2 T) d -> b h k2 T k1 d", k1=ks1, k2=ks2)
        k_h = self.rotary_emb(k_h, scale=scale)
        k_h = rearrange(k_h, "b h k2 T k1 d -> b h (k1 k2 T) d", k1=ks1, k2=ks2)
        # width
        k_w = rearrange(k_w, "b h (k1 k2 T) d -> b h T k1 k2 d", k1=ks1, k2=ks2)
        k_w = self.rotary_emb(k_w, scale=scale)
        k_w = rearrange(k_w, "b h T k1 k2 d -> b h (k1 k2 T) d", k1=ks1, k2=ks2)
        # combine
        k = torch.cat([k_t, k_h, k_w], dim=-1)
        return k

    def forward(self, x: torch.Tensor, scale=None, H=None, W=None) -> torch.Tensor:
        attn_mask = None
        if self.kernel_size is not None:
            B, *dims, C = x.shape
            assert len(dims) == len(
                self.kernel_size
            ), f"input shape and kernel size mismatch, {dims} vs {self.kernel_size}"
            kernel_size = [(k if k >= 0 else s) for s, k in zip(dims, self.kernel_size)]
            N = np.prod(kernel_size)
            indices = list(range(1, x.dim() - 1))

            x, padding_mask = pad_to_multiples(x, kernel_size, indices, extra_pad_on_dims=self.extra_pad_on_dims)
            num_splits = [x.size(dim) // k for dim, k in zip(indices, kernel_size)]
            x = split_seq_cat_batch(x, kernel_size, indices)
            qkv_b = x.shape[0]
            if padding_mask is not None:
                attn_mask = padding_mask.narrow(-1, 0, 1).squeeze(-1)
                attn_mask = 1.0 - split_seq_cat_batch(attn_mask, kernel_size, indices).to(x.dtype)
                attn_mask.masked_fill_(attn_mask.bool(), float("-inf"))
                attn_mask = attn_mask.view(qkv_b, N)
                attn_mask = attn_mask[:, None, :].expand(qkv_b, N, N).unsqueeze(1)
        else:
            B, N, C = x.shape
            qkv_b = B

        qkv = self.qkv(x)
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (N > 128) and self.kernel_size is None
        enable_sdpa = self.enable_flash_attn and (N > 128) and self.kernel_size is not None
        qkv_shape = (qkv_b, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.qk_norm_legacy:
            # WARNING: this may be a bug
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                if self.kernel_size is not None:
                    q = self.rotate_3d(q, self.kernel_size[0], self.kernel_size[1], scale=scale)
                    k = self.rotate_3d(k, self.kernel_size[0], self.kernel_size[1], scale=scale)
                elif H is not None and W is not None:
                    q = self.rotate_3d(q, H, W, scale=scale)
                    k = self.rotate_3d(k, H, W, scale=scale)
                else:
                    q = self.rotary_emb(q)
                    k = self.rotary_emb(k)

        if enable_flash_attn:
            from flash_attn import flash_attn_func

            # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=self.is_causal,
            )
        elif enable_sdpa:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                scale=self.scale,
                is_causal=self.is_causal,
            )
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            if self.is_causal:
                causal_mask = torch.tril(torch.ones_like(attn), diagonal=0)
                causal_mask = torch.where(causal_mask.bool(), 0, float("-inf"))
                attn += causal_mask
            if attn_mask is not None:
                attn += attn_mask
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (qkv_b, N, C)
        if not enable_flash_attn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self.kernel_size is not None:
            x = x.reshape(qkv_b, *kernel_size, C)
            x = split_batch_cat_seq(x, B, num_splits, indices)
            x = remove_padding_nd(x, dims, indices)
            assert x.shape == (B, *dims, C)
        return x


class KVCompressAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        sampling="conv",
        sr_ratio=1,
        mem_eff_attention=False,
        attn_half=False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.sr_ratio = sr_ratio
        self.sampling = sampling
        if sr_ratio > 1 and sampling == "conv":
            # Avg Conv Init.
            self.sr = nn.Conv2d(dim, dim, groups=dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr.weight.data.fill_(1 / sr_ratio**2)
            self.sr.bias.data.zero_()
            self.norm = nn.LayerNorm(dim)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mem_eff_attention = mem_eff_attention
        self.attn_half = attn_half

    def downsample_2d(self, tensor, H, W, scale_factor, sampling=None):
        if sampling is None or scale_factor == 1:
            return tensor
        B, N, C = tensor.shape

        if sampling == "uniform_every":
            return tensor[:, ::scale_factor], int(N // scale_factor)

        tensor = tensor.reshape(B, H, W, C).permute(0, 3, 1, 2)
        new_H, new_W = int(H / scale_factor), int(W / scale_factor)
        new_N = new_H * new_W

        if sampling == "ave":
            tensor = F.interpolate(tensor, scale_factor=1 / scale_factor, mode="nearest").permute(0, 2, 3, 1)
        elif sampling == "uniform":
            tensor = tensor[:, :, ::scale_factor, ::scale_factor].permute(0, 2, 3, 1)
        elif sampling == "conv":
            tensor = self.sr(tensor).reshape(B, C, -1).permute(0, 2, 1)
            tensor = self.norm(tensor)
        else:
            raise ValueError

        return tensor.reshape(B, new_N, C).contiguous(), new_N

    def forward(self, x: torch.Tensor, mask=None, HW=None, block_id=None, **kwargs) -> torch.Tensor:
        B, N, C = x.shape
        new_N = N
        H, W = HW
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (N > B)

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype
        # KV compression
        if self.sr_ratio > 1:
            k, new_N = self.downsample_2d(k, H, W, self.sr_ratio, sampling=self.sampling)
            v, new_N = self.downsample_2d(v, H, W, self.sr_ratio, sampling=self.sampling)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        k = k.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)

        q, k = self.q_norm(q), self.k_norm(k)

        if enable_flash_attn:
            from flash_attn import flash_attn_func

            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
            )

        elif self.mem_eff_attention:
            attn_bias = None
            if mask is not None:
                attn_bias = torch.zeros([B * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
                attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float("-inf"))
            x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        else:
            # (B, N, #heads, #dim) -> (B, #heads, N, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            if not self.attn_half:
                attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (B, N, C)
        if not enable_flash_attn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SeqParallelAttention(Attention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        rope=None,
        qk_norm_legacy: bool = False,
        kernel_size: Optional[Sequence[int]] = None,
        shift_window: bool = False,
        temporal: bool = False,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            enable_flash_attn=enable_flash_attn,
            rope=rope,
            qk_norm_legacy=qk_norm_legacy,
            kernel_size=kernel_size,
            shift_window=shift_window,
        )
        self.temporal = temporal

    def forward(self, x: torch.Tensor, scale=None, H=None, W=None) -> torch.Tensor:
        sp_group = get_sequence_parallel_group()
        sp_size = dist.get_world_size(sp_group)
        qkv = self.qkv(x)

        if self.kernel_size is not None:
            B, *dims, C = x.shape
            N = np.prod(dims)
            qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
            qkv = qkv.view(qkv_shape)

            # apply all_to_all to gather sequence and split attention heads
            # [B, SUB_N, 3, NUM_HEAD, HEAD_DIM] -> [B, N, 3, NUM_HEAD_PER_DEVICE, HEAD_DIM]
            qkv = all_to_all(qkv, sp_group, scatter_dim=3, gather_dim=1)

            dims_sp = [dims[0] * sp_size, *dims[1:]]
            C_sp = C // sp_size
            qkv = qkv.view([qkv.shape[0], *dims_sp, 3 * C_sp])

            assert len(dims_sp) == len(
                self.kernel_size
            ), f"input shape and kernel size mismatch, {dims_sp} vs {self.kernel_size}"
            kernel_size = [(k if k >= 0 else s) for s, k in zip(dims_sp, self.kernel_size)]

            indices = list(range(1, x.dim() - 1))
            qkv, qkv_padding_mask = pad_to_multiples(
                qkv, kernel_size, indices, extra_pad_on_dims=self.extra_pad_on_dims
            )
            num_splits = [qkv.size(dim) // k for dim, k in zip(indices, kernel_size)]

            qkv = split_seq_cat_batch(qkv, kernel_size, indices)
            qkv_b = qkv.shape[0]
            qkv_n = np.prod(kernel_size)

            if qkv_padding_mask is not None:
                qkv_padding_mask = qkv_padding_mask.narrow(-1, 0, 1).squeeze(-1)
                qkv_padding_mask = split_seq_cat_batch(qkv_padding_mask, kernel_size, indices).to(qkv.dtype)
                qkv_padding_mask.masked_fill_(qkv_padding_mask.logical_not(), float("-inf"))
                qkv_padding_mask = qkv_padding_mask.view(qkv_b, qkv_n)
                qkv_padding_mask = qkv_padding_mask[:, None, :].expand(qkv_b, qkv_n, qkv_n).unsqueeze(1)

            qkv_shape = (qkv_b, qkv_n, 3, self.num_heads // sp_size, self.head_dim)
            qkv = qkv.view(qkv_shape)
        else:
            B, N, C = x.shape
            qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
            qkv = qkv.view(qkv_shape)
            if not self.temporal:
                # apply all_to_all to gather sequence and split attention heads
                # [B, SUB_N, 3, NUM_HEAD, HEAD_DIM] -> [B, N, 3, NUM_HEAD_PER_DEVICE, HEAD_DIM]
                qkv = all_to_all(qkv, sp_group, scatter_dim=3, gather_dim=1)
            else:
                sp_size = 1
            qkv_b, qkv_n = qkv.shape[0], qkv.shape[1]
            qkv_padding_mask = None

        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (N > qkv_b) and self.kernel_size is None
        enable_sdpa = self.enable_flash_attn and (N > qkv_b) and self.kernel_size is not None

        qkv = qkv.permute(2, 0, 3, 1, 4)
        # ERROR: Should qk_norm first
        q, k, v = qkv.unbind(0)
        if self.qk_norm_legacy:
            # WARNING: this may be a bug
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                if self.kernel_size is not None:
                    q = self.rotate_3d(q, self.kernel_size[0], self.kernel_size[1], scale=scale)
                    k = self.rotate_3d(k, self.kernel_size[0], self.kernel_size[1], scale=scale)
                elif H is not None and W is not None:
                    if self.temporal is False and self.kernel_size is None:
                        H = H * sp_size
                    q = self.rotate_3d(q, H, W, scale=scale)
                    k = self.rotate_3d(k, H, W, scale=scale)
                else:
                    q = self.rotary_emb(q)
                    k = self.rotary_emb(k)

        if enable_flash_attn:
            from flash_attn import flash_attn_func

            # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=self.is_causal,
            )
        elif enable_sdpa:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=qkv_padding_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                scale=self.scale,
                is_causal=self.is_causal,
            )
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            if self.is_causal:
                causal_mask = torch.tril(torch.ones_like(attn), diagonal=0)
                causal_mask = torch.where(causal_mask.bool(), 0, float("-inf"))
                attn += causal_mask
            if qkv_padding_mask is not None:
                attn += qkv_padding_mask
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        if not enable_flash_attn:
            x = x.transpose(1, 2)  # (B, #heads, N, #dim) -> (B, N, #heads, #dim)

        if self.kernel_size is not None:
            x = x.reshape(qkv_b, *kernel_size, C_sp)
            x = split_batch_cat_seq(x, B, num_splits, indices)
            x = remove_padding_nd(x, dims_sp, indices)

            x_shape = (B, N * sp_size, self.num_heads // sp_size, self.head_dim)
            x = x.reshape(x_shape)
            # apply all to all to gather back attention heads and split sequence
            # [B, N, NUM_HEAD_PER_DEVICE, HEAD_DIM]  -> [B, SUB_N, NUM_HEAD, HEAD_DIM]
            x = all_to_all(x, sp_group, scatter_dim=1, gather_dim=2)

            x = x.view(B, *dims, C)

            assert x.shape == (B, *dims, C)
        else:
            if not self.temporal:
                # apply all to all to gather back attention heads and split sequence
                # [B, N, NUM_HEAD_PER_DEVICE, HEAD_DIM]  -> [B, SUB_N, NUM_HEAD, HEAD_DIM]
                x = all_to_all(x, sp_group, scatter_dim=1, gather_dim=2)

            x_output_shape = (B, N, C)
            x = x.reshape(x_output_shape)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        if mask is None:
            Bc, Nc, _ = cond.shape
            assert Bc == B
            mask = [Nc] * B

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)

        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiHeadCrossAttentionForCondition(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(cond).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(x).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens(mask, [N] * B)
        cond = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)

        cond = cond.view(1, -1, C)
        cond = self.proj(cond)
        cond = self.proj_drop(cond)
        return cond


class SeqParallelMultiHeadCrossAttention(MultiHeadCrossAttention):
    def __init__(
        self,
        d_model,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        sp_group = get_sequence_parallel_group()
        sp_size = dist.get_world_size(sp_group)
        B, SUB_N, C = x.shape  # [B, TS/p, C]
        N = SUB_N * sp_size

        if mask is None:
            Bc, Nc, _ = cond.shape
            assert Bc == B
            mask = [Nc] * B

        # shape:
        # q, k, v: [B, SUB_N, NUM_HEADS, HEAD_DIM]
        q = self.q_linear(x).view(B, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        kv = split_forward_gather_backward(kv, get_sequence_parallel_group(), dim=3, grad_scale="down")
        k, v = kv.unbind(2)

        # apply all_to_all to gather sequence and split attention heads
        q = all_to_all(q, sp_group, scatter_dim=2, gather_dim=1)

        q = q.view(1, -1, self.num_heads // sp_size, self.head_dim)
        k = k.view(1, -1, self.num_heads // sp_size, self.head_dim)
        v = v.view(1, -1, self.num_heads // sp_size, self.head_dim)

        # compute attention
        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)

        # apply all to all to gather back attention heads and scatter sequence
        x = x.view(B, -1, self.num_heads // sp_size, self.head_dim)
        x = all_to_all(x, sp_group, scatter_dim=1, gather_dim=2)

        # apply output projection
        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, num_patch, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final, x, shift, scale)
        x = self.linear(x)
        return x


class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, num_patch, out_channels, d_t=None, d_s=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)
        self.out_channels = out_channels
        self.d_t = d_t
        self.d_s = d_s

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(self, x, t, x_mask=None, t0=None, T=None, S=None):
        if T is None:
            T = self.d_t
        if S is None:
            S = self.d_s
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        if x_mask is not None:
            shift_zero, scale_zero = (self.scale_shift_table[None] + t0[:, None]).chunk(2, dim=1)
            x_zero = t2i_modulate(self.norm_final(x), shift_zero, scale_zero)
            x = self.t_mask_select(x_mask, x, x_zero, T, S)
        x = self.linear(x)
        return x


# ===============================================
# Embedding Layers for Timesteps and Class Labels
# ===============================================


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]).cuda() < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class SizeEmbedder(TimestepEmbedder):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

    def forward(self, s, bs):
        if s.ndim == 1:
            s = s[:, None]
        assert s.ndim == 2
        if s.shape[0] != bs:
            s = s.repeat(bs // s.shape[0], 1)
            assert s.shape[0] == bs
        b, dims = s.shape[0], s.shape[1]
        s = rearrange(s, "b d -> (b d)")
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(self.dtype)
        s_emb = self.mlp(s_freq)
        s_emb = rearrange(s_emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return s_emb

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        uncond_prob,
        act_layer=nn.GELU(approximate="tanh"),
        token_num=120,
    ):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            drop=0,
        )
        self.register_buffer(
            "y_embedding",
            torch.randn(token_num, in_channels) / in_channels**0.5,
        )
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None, uncond_prob=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        uncond_prob = uncond_prob if uncond_prob is not None else self.uncond_prob
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None, uncond_prob=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        uncond_prob = uncond_prob if uncond_prob is not None else self.uncond_prob
        use_dropout = uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids=force_drop_ids, uncond_prob=uncond_prob)
        caption = self.y_proj(caption)
        return caption


class PositionEmbedding2D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, "dim must be divisible by 4"
        half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_sin_cos_emb(self, t: torch.Tensor):
        out = torch.einsum("i,d->id", t, self.inv_freq)
        emb_cos = torch.cos(out)
        emb_sin = torch.sin(out)
        return torch.cat((emb_sin, emb_cos), dim=-1)

    @functools.lru_cache(maxsize=512)
    def _get_cached_emb(
        self,
        device: torch.device,
        dtype: torch.dtype,
        h: int,
        w: int,
        scale: float = 1.0,
        base_size: Optional[int] = None,
    ):
        grid_h = torch.arange(h, device=device) / scale
        grid_w = torch.arange(w, device=device) / scale
        if base_size is not None:
            grid_h *= base_size / h
            grid_w *= base_size / w
        grid_h, grid_w = torch.meshgrid(
            grid_w,
            grid_h,
            indexing="ij",
        )  # here w goes first
        grid_h = grid_h.t().reshape(-1)
        grid_w = grid_w.t().reshape(-1)
        emb_h = self._get_sin_cos_emb(grid_h)
        emb_w = self._get_sin_cos_emb(grid_w)
        return torch.concat([emb_h, emb_w], dim=-1).unsqueeze(0).to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
        scale: Optional[float] = 1.0,
        base_size: Optional[int] = None,
    ) -> torch.Tensor:
        return self._get_cached_emb(x.device, x.dtype, h, w, scale, base_size)


# ===============================================
# Sine/Cosine Positional Embedding Functions
# ===============================================
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, scale=1.0, base_size=None):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / scale
    if base_size is not None:
        grid_h *= base_size / grid_size[0]
        grid_w *= base_size / grid_size[1]
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
