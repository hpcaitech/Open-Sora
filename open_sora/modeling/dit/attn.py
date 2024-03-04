from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from colossalai.logging import get_dist_logger
from colossalai.shardformer.layer._operation import gather_forward_split_backward

from open_sora.utils.comm import all_to_all, async_all_gather_proj_for_two


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
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.scale = head_dim**-0.5
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sdpa = sdpa

        self.to_q = nn.Linear(query_dim, self.hidden_size, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, self.hidden_size, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, self.hidden_size, bias=bias)

        self.to_out = nn.Sequential(nn.Linear(self.hidden_size, query_dim), nn.Dropout(dropout))

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
        value = value.view(bsz, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if mask is not None:
            assert mask.shape == (bsz, 1, q_len, kv_seq_len)
        if self.sdpa:
            attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=self.scale)
        else:
            attn_weights = torch.matmul(query, key.transpose(2, 3)) / self.scale
            assert attn_weights.shape == (bsz, self.num_heads, q_len, kv_seq_len)
            if mask is not None:
                attn_weights = attn_weights + mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_output = torch.matmul(attn_weights, value)
        assert attn_output.shape == (bsz, self.num_heads, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.to_out(attn_output)
        return attn_output


class SeqParallelCrossAttention(CrossAttention):
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
        super().__init__(
            query_dim,
            cross_attention_dim,
            num_heads,
            head_dim,
            dropout,
            bias,
            sdpa,
        )
        self.seq_parallel_group = seq_parallel_group
        self.seq_parallel_size = dist.get_world_size(self.seq_parallel_group) if seq_parallel_group is not None else 1
        assert self.num_heads % self.seq_parallel_size == 0

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
        if self.seq_parallel_size > 1:
            query = all_to_all(query, self.seq_parallel_group, scatter_dim=2, gather_dim=1)
            key = all_to_all(key, self.seq_parallel_group, scatter_dim=2, gather_dim=1)
            value = all_to_all(value, self.seq_parallel_group, scatter_dim=2, gather_dim=1)

        q_len *= self.seq_parallel_size
        kv_seq_len *= self.seq_parallel_size

        # [B, S, H/P] -> [B, S, N/P, D] -> [B, N/P, S, D]
        query = query.view(bsz, q_len, num_heads_parallel, self.head_dim).transpose(1, 2)
        key = key.view(bsz, kv_seq_len, num_heads_parallel, self.head_dim).transpose(1, 2)
        value = value.view(bsz, kv_seq_len, num_heads_parallel, self.head_dim).transpose(1, 2)

        if mask is not None:
            assert mask.shape == (bsz, 1, q_len, kv_seq_len)
        if self.sdpa:
            attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=self.scale)
        else:
            attn_weights = torch.matmul(query, key.transpose(2, 3)) / self.scale
            assert attn_weights.shape == (bsz, num_heads_parallel, q_len, kv_seq_len)
            if mask is not None:
                attn_weights = attn_weights + mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_output = torch.matmul(attn_weights, value)
        assert attn_output.shape == (bsz, num_heads_parallel, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, hidden_size_parallel)
        # [B, S, H/P] -> [B, S/P, H]
        if self.seq_parallel_size > 1:
            attn_output = all_to_all(attn_output, self.seq_parallel_group, scatter_dim=1, gather_dim=2)
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
        self.seq_parallel_rank = dist.get_rank(self.seq_parallel_group) if seq_parallel_group is not None else 0
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
        bias = bias = proj_layer.bias[self.sequence_parallel_param_slice] if proj_layer.bias is not None else None
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
                hidden_states = gather_forward_split_backward(hidden_states, 1, self.seq_parallel_group)
                context = gather_forward_split_backward(context, 1, self.seq_parallel_group)
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
        query = query.view(bsz, q_len, num_heads_parallel, self.head_dim).transpose(1, 2)
        key = key.view(bsz, kv_seq_len, num_heads_parallel, self.head_dim).transpose(1, 2)
        value = value.view(bsz, kv_seq_len, num_heads_parallel, self.head_dim).transpose(1, 2)

        if mask is not None:
            assert mask.shape == (bsz, 1, q_len, kv_seq_len)
        if self.sdpa:
            attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=self.scale)
        else:
            attn_weights = torch.matmul(query, key.transpose(2, 3)) / self.scale
            assert attn_weights.shape == (bsz, num_heads_parallel, q_len, kv_seq_len)
            if mask is not None:
                attn_weights = attn_weights + mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_output = torch.matmul(attn_weights, value)
        assert attn_output.shape == (bsz, num_heads_parallel, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, hidden_size_parallel)
        # [B, S, H/P] -> [B, S/P, H]
        if self.seq_parallel_size > 1:
            attn_output = all_to_all(attn_output, self.seq_parallel_group, scatter_dim=1, gather_dim=2)
        attn_output = self.to_out(attn_output)
        return attn_output
