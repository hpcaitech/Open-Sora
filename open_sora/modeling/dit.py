# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from colossalai.logging import get_dist_logger
from colossalai.shardformer.layer._operation import gather_forward_split_backward
from timm.models.vision_transformer import Mlp

from open_sora.utils.comm import (
    all_to_all,
    async_all_gather_proj_for_two,
    gather_seq,
    split_seq,
)


class CrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the context. If not given, defaults to `query_dim`.
        num_heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        head_dim (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        bias=False,
        sdpa=True,
    ):
        super().__init__()
        self.hidden_size = head_dim * num_heads
        cross_attention_dim = (
            cross_attention_dim if cross_attention_dim is not None else query_dim
        )

        self.scale = head_dim**-0.5
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sdpa = sdpa

        self.to_q = nn.Linear(query_dim, self.hidden_size, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, self.hidden_size, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, self.hidden_size, bias=bias)

        self.to_out = nn.Sequential(
            nn.Linear(self.hidden_size, query_dim), nn.Dropout(dropout)
        )

    def forward(self, hidden_states, context=None, mask=None):
        bsz, q_len, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        kv_seq_len = context.shape[1]
        key = self.to_k(context)
        value = self.to_v(context)

        # [B, S, H, D]
        query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, kv_seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        if mask is not None:
            assert mask.shape == (bsz, 1, q_len, kv_seq_len)
        if self.sdpa:
            attn_output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, scale=self.scale
            )
        else:
            attn_weights = torch.matmul(query, key.transpose(2, 3)) / self.scale
            assert attn_weights.shape == (bsz, self.num_heads, q_len, kv_seq_len)
            if mask is not None:
                attn_weights = attn_weights + mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            attn_output = torch.matmul(attn_weights, value)
        assert attn_output.shape == (bsz, self.num_heads, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.to_out(attn_output)
        return attn_output


class SeqParallelCrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the context. If not given, defaults to `query_dim`.
        num_heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        head_dim (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        bias=False,
        sdpa=True,
        seq_parallel_group=None,
    ):
        super().__init__()
        self.hidden_size = head_dim * num_heads
        cross_attention_dim = (
            cross_attention_dim if cross_attention_dim is not None else query_dim
        )

        self.scale = head_dim**-0.5
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sdpa = sdpa
        self.seq_parallel_group = seq_parallel_group
        self.seq_parallel_size = (
            dist.get_world_size(self.seq_parallel_group)
            if seq_parallel_group is not None
            else 1
        )
        assert self.num_heads % self.seq_parallel_size == 0

        self.to_q = nn.Linear(query_dim, self.hidden_size, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, self.hidden_size, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, self.hidden_size, bias=bias)

        self.to_out = nn.Sequential(
            nn.Linear(self.hidden_size, query_dim), nn.Dropout(dropout)
        )

    def forward(self, hidden_states, context=None, mask=None):
        bsz, q_len, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        kv_seq_len = context.shape[1]
        key = self.to_k(context)
        value = self.to_v(context)

        # [B, S/P, H] -> [B, S, H/P]
        num_heads_parallel = self.num_heads // self.seq_parallel_size
        hidden_size_parallel = self.hidden_size // self.seq_parallel_size
        if self.seq_parallel_group is not None and self.seq_parallel_size > 1:
            query = all_to_all(
                query, self.seq_parallel_group, scatter_dim=2, gather_dim=1
            )
            key = all_to_all(key, self.seq_parallel_group, scatter_dim=2, gather_dim=1)
            value = all_to_all(
                value, self.seq_parallel_group, scatter_dim=2, gather_dim=1
            )

        q_len *= self.seq_parallel_size
        kv_seq_len *= self.seq_parallel_size

        # [B, S, H/P] -> [B, S, N/P, D] -> [B, N/P, S, D]
        query = query.view(bsz, q_len, num_heads_parallel, self.head_dim).transpose(
            1, 2
        )
        key = key.view(bsz, kv_seq_len, num_heads_parallel, self.head_dim).transpose(
            1, 2
        )
        value = value.view(
            bsz, kv_seq_len, num_heads_parallel, self.head_dim
        ).transpose(1, 2)

        if mask is not None:
            assert mask.shape == (bsz, 1, q_len, kv_seq_len)
        if self.sdpa:
            attn_output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, scale=self.scale
            )
        else:
            attn_weights = torch.matmul(query, key.transpose(2, 3)) / self.scale
            assert attn_weights.shape == (bsz, num_heads_parallel, q_len, kv_seq_len)
            if mask is not None:
                attn_weights = attn_weights + mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            attn_output = torch.matmul(attn_weights, value)
        assert attn_output.shape == (bsz, num_heads_parallel, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, hidden_size_parallel)
        # [B, S, H/P] -> [B, S/P, H]
        if self.seq_parallel_group is not None and self.seq_parallel_size > 1:
            attn_output = all_to_all(
                attn_output, self.seq_parallel_group, scatter_dim=1, gather_dim=2
            )
        attn_output = self.to_out(attn_output)
        return attn_output


class FastSeqParallelCrossAttention(SeqParallelCrossAttention):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0,
        bias=False,
        sdpa=True,
        seq_parallel_group=None,
        overlap=False,
    ):
        super().__init__(
            query_dim,
            cross_attention_dim,
            num_heads,
            head_dim,
            dropout,
            bias,
            sdpa,
            seq_parallel_group,
        )
        self.seq_parallel_rank = (
            dist.get_rank(self.seq_parallel_group)
            if seq_parallel_group is not None
            else 0
        )
        self.sequence_parallel_param_slice = slice(
            self.hidden_size // self.seq_parallel_size * self.seq_parallel_rank,
            self.hidden_size // self.seq_parallel_size * (self.seq_parallel_rank + 1),
        )
        if overlap and self.seq_parallel_size != 2:
            logger = get_dist_logger()
            logger.warning(
                "FastSeqParallelCrossAttention only supports overlap with seq_parallel_size=2. Fallback to non-overlap",
                ranks=[0],
            )
            overlap = False
        self.overlap = overlap

    def _get_sliced_params(self, proj_layer: nn.Linear):
        bias = bias = (
            proj_layer.bias[self.sequence_parallel_param_slice]
            if proj_layer.bias is not None
            else None
        )
        return proj_layer.weight[self.sequence_parallel_param_slice], bias

    def _proj(self, x: torch.Tensor, proj_layer: nn.Linear):
        return F.linear(
            x,
            *self._get_sliced_params(proj_layer),
        )

    def forward(self, hidden_states, context=None, mask=None):
        bsz, q_len, _ = hidden_states.shape

        context = context if context is not None else hidden_states
        kv_seq_len = context.shape[1]
        if self.seq_parallel_size > 1:
            if self.overlap and self.seq_parallel_size == 2:
                query, key, value = async_all_gather_proj_for_two(
                    hidden_states,
                    context,
                    *self._get_sliced_params(self.to_q),
                    *self._get_sliced_params(self.to_k),
                    *self._get_sliced_params(self.to_v),
                    dim=1,
                    process_group=self.seq_parallel_group,
                    sp_size=self.seq_parallel_size,
                    sp_rank=self.seq_parallel_rank,
                )
            else:
                # [B, S/P, H] -> [B, S, H]
                hidden_states = gather_forward_split_backward(
                    hidden_states, 1, self.seq_parallel_group
                )
                context = gather_forward_split_backward(
                    context, 1, self.seq_parallel_group
                )
                query = self._proj(hidden_states, self.to_q)
                key = self._proj(context, self.to_k)
                value = self._proj(context, self.to_v)
        else:
            query = self.to_q(hidden_states)
            key = self.to_k(context)
            value = self.to_v(context)
        # output is [B, S, H/P]

        num_heads_parallel = self.num_heads // self.seq_parallel_size
        hidden_size_parallel = self.hidden_size // self.seq_parallel_size
        q_len *= self.seq_parallel_size
        kv_seq_len *= self.seq_parallel_size

        # [B, S, H/P] -> [B, S, N/P, D] -> [B, N/P, S, D]
        query = query.view(bsz, q_len, num_heads_parallel, self.head_dim).transpose(
            1, 2
        )
        key = key.view(bsz, kv_seq_len, num_heads_parallel, self.head_dim).transpose(
            1, 2
        )
        value = value.view(
            bsz, kv_seq_len, num_heads_parallel, self.head_dim
        ).transpose(1, 2)

        if mask is not None:
            assert mask.shape == (bsz, 1, q_len, kv_seq_len)
        if self.sdpa:
            attn_output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, scale=self.scale
            )
        else:
            attn_weights = torch.matmul(query, key.transpose(2, 3)) / self.scale
            assert attn_weights.shape == (bsz, num_heads_parallel, q_len, kv_seq_len)
            if mask is not None:
                attn_weights = attn_weights + mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            attn_output = torch.matmul(attn_weights, value)
        assert attn_output.shape == (bsz, num_heads_parallel, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, hidden_size_parallel)
        # [B, S, H/P] -> [B, S/P, H]
        if self.seq_parallel_size > 1:
            attn_output = all_to_all(
                attn_output, self.seq_parallel_group, scatter_dim=1, gather_dim=2
            )
        attn_output = self.to_out(attn_output)
        return attn_output


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


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
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            self.mlp[0].weight.dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class PatchEmbedder(nn.Module):
    """Patch Embedding Layer for flat 4D video tensors."""

    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, S, C, P, P] -> [B, S, C*P*P]
        x = x.view(*x.shape[:2], -1)
        out = F.linear(
            x, self.proj.weight.view(self.proj.weight.shape[0], -1), self.proj.bias
        )
        out = self.norm(out)
        # [B, S, H]
        return out


class TextEmbedder(nn.Module):
    def __init__(
        self,
        in_features: int,
        embed_dim: int = 768,
        bias: bool = True,
        dropout_prob: float = 0.0,
        use_proj: bool = True,
    ) -> None:
        super().__init__()
        self.dropout_prob = dropout_prob
        self.use_proj = use_proj
        if self.use_proj:
            self.proj = nn.Linear(in_features, embed_dim, bias=bias)

    def drop_sample(self, x: torch.Tensor) -> torch.Tensor:
        drop_ids = torch.rand(x.shape[0], 1, 1, device=x.device) < self.dropout_prob
        x = torch.where(drop_ids, torch.zeros_like(x), x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_dropout = self.dropout_prob > 0
        if self.training and use_dropout:
            x = self.drop_sample(x)
        if self.use_proj:
            # [B, S, C] -> [B, S, H]
            x = self.proj(x)
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings=262114) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self._set_pos_embed_cache(max_position_embeddings)

    def _set_pos_embed_cache(self, seq_len: int, device="cpu", dtype=torch.float):
        self.max_seq_len_cached = seq_len
        pos_embed = get_2d_sincos_pos_embed(self.dim, math.ceil(seq_len**0.5))
        pos_embed = torch.from_numpy(pos_embed).to(device=device, dtype=dtype)
        # [S, H]
        self.register_buffer("pos_embed_cache", pos_embed, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, S, H]
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len_cached:
            self._set_pos_embed_cache(seq_len, x.device, x.dtype)
        pos_embed = self.pos_embed_cache[None, :seq_len]
        return pos_embed


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        cross_attention_dim=None,
        mlp_ratio=4.0,
        seq_parallel_group=None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SeqParallelCrossAttention(
            query_dim=hidden_size,
            cross_attention_dim=cross_attention_dim,
            num_heads=num_heads,
            head_dim=hidden_size // num_heads,
            bias=True,
            sdpa=True,
            seq_parallel_group=seq_parallel_group,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, attention_mask, t, context=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(t).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), context, attention_mask
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.patch_size = patch_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def unpatchify(self, x):
        b, s, h = x.shape
        return x.view(b, s, -1, self.patch_size, self.patch_size)

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        x = self.unpatchify(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        patch_size=2,
        in_channels=3,
        text_embed_dim=512,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        max_num_embeddings=256 * 1024,
        text_dropout_prob=0.1,
        learn_sigma=True,
        use_cross_attn=True,
        seq_parallel_group=None,
    ):
        super().__init__()
        self.grad_checkpointing = False
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.seq_parallel_group = seq_parallel_group
        self.seq_parallel_size = (
            dist.get_world_size(self.seq_parallel_group)
            if seq_parallel_group is not None
            else 1
        )
        self.seq_parallel_rank = (
            dist.get_rank(self.seq_parallel_group)
            if seq_parallel_group is not None
            else 0
        )

        self.video_embedder = PatchEmbedder(
            patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.pos_embed = PositionEmbedding(hidden_size, max_num_embeddings)
        self.text_embedder = TextEmbedder(
            text_embed_dim,
            hidden_size,
            bias=True,
            dropout_prob=text_dropout_prob,
            use_proj=not use_cross_attn,
        )
        if not use_cross_attn:
            cross_attn_dim = None
        else:
            cross_attn_dim = text_embed_dim
        self.use_cross_attn = use_cross_attn

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    cross_attn_dim,
                    mlp_ratio=mlp_ratio,
                    seq_parallel_group=seq_parallel_group,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize text embedding layer
        if self.text_embedder.use_proj:
            nn.init.normal_(self.text_embedder.proj.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _prepare_mask(self, attention_mask: Optional[torch.Tensor], dtype: torch.dtype):
        if attention_mask is not None:
            assert attention_mask.ndim == 4
            attention_mask = attention_mask.to(dtype)
            inverted_mask = 1.0 - attention_mask
            return inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(dtype).min
            )
        return attention_mask

    def enable_gradient_checkpointing(self):
        self.grad_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.grad_checkpointing = False

    def forward(
        self,
        video_latent_states,
        t,
        text_latent_states=None,
        attention_mask=None,
    ):
        """
        video_latent_states: [B, S, C, P, P]
        """
        video_latent_states = self.video_embedder(video_latent_states)
        text_len = text_latent_states.shape[1]
        text_latent_states = self.text_embedder(text_latent_states)
        if not self.use_cross_attn:
            video_latent_states = torch.cat(
                [text_latent_states, video_latent_states], dim=1
            )
            text_latent_states = None
        pos_embed = self.pos_embed(video_latent_states)
        video_latent_states = video_latent_states + pos_embed
        t = self.t_embedder(t)  # (N, D)
        attention_mask = self._prepare_mask(attention_mask, video_latent_states.dtype)

        if self.seq_parallel_group is not None and self.seq_parallel_size > 1:
            assert video_latent_states.shape[1] % self.seq_parallel_size == 0
            video_latent_states = split_seq(
                video_latent_states, self.seq_parallel_size, self.seq_parallel_rank
            )
            if text_latent_states is not None:
                assert text_latent_states.shape[1] % self.seq_parallel_size == 0
                text_latent_states = split_seq(
                    text_latent_states, self.seq_parallel_size, self.seq_parallel_rank
                )

        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                video_latent_states = torch.utils.checkpoint.checkpoint(
                    block,
                    video_latent_states,
                    attention_mask,
                    t,
                    text_latent_states,
                )
            else:
                video_latent_states = block(
                    video_latent_states, attention_mask, t, text_latent_states
                )

        if self.seq_parallel_group is not None and self.seq_parallel_size > 1:
            video_latent_states = gather_seq(
                video_latent_states,
                self.seq_parallel_size,
                self.seq_parallel_rank,
                self.seq_parallel_group,
            )

        if not self.use_cross_attn:
            video_latent_states = video_latent_states[:, text_len:]
        video_latent_states = self.final_layer(video_latent_states, t)
        return video_latent_states

    def forward_with_cfg(
        self, x, t, text_latent_states, cfg_scale, attention_mask=None
    ):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(
            combined, t, text_latent_states, attention_mask=attention_mask
        )
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        if self.learn_sigma:
            c = model_out.shape[2]
            assert c == 2 * self.in_channels
            eps, rest = model_out.chunk(2, dim=2)
        else:
            eps = model_out
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        if self.learn_sigma:
            return torch.cat([eps, rest], dim=2)
        return eps


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


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


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
    "DiT-XL/8": DiT_XL_8,
    "DiT-L/2": DiT_L_2,
    "DiT-L/4": DiT_L_4,
    "DiT-L/8": DiT_L_8,
    "DiT-B/2": DiT_B_2,
    "DiT-B/4": DiT_B_4,
    "DiT-B/8": DiT_B_8,
    "DiT-S/2": DiT_S_2,
    "DiT-S/4": DiT_S_4,
    "DiT-S/8": DiT_S_8,
}
