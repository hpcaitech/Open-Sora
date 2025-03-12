from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.shardformer.layer import (FusedLinear1D_Col, FusedLinear1D_Row,
                                          Linear1D_Col, Linear1D_Row)
from colossalai.shardformer.layer._operation import all_to_all_comm
from colossalai.shardformer.layer.attn import RingComm, _rescale_out_lse
from colossalai.shardformer.layer.utils import is_share_sp_tp
from colossalai.shardformer.policies.base_policy import (
    ModulePolicyDescription, Policy, SubModuleReplacementDescription)
from colossalai.shardformer.shard import ShardConfig
from einops import rearrange
from flash_attn.flash_attn_interface import (_flash_attn_backward,
                                             _flash_attn_forward)
from liger_kernel.ops.rope import LigerRopeFunction

try:
    from flash_attn_interface import \
        _flash_attn_backward as _flash_attn_backward_v3
    from flash_attn_interface import \
        _flash_attn_forward as _flash_attn_forward_v3

    SUPPORT_FA3 = True
except:
    SUPPORT_FA3 = False

from torch import Tensor

from opensora.acceleration.checkpoint import auto_grad_checkpoint

from .layers import DoubleStreamBlock, SingleStreamBlock
from .math import apply_rope, attention
from .model import MMDiTModel


class _SplitForwardGatherBackwardVarLen(torch.autograd.Function):
    """
    Split the input and keep only the corresponding chuck to the rank.

    Args:
        input_ (`torch.Tensor`): input matrix.
        dim (int): the dimension to perform split and gather
        process_group (`torch.distributed.ProcessGroup`): the process group used for collective communication

    """

    @staticmethod
    def forward(ctx, input_, dim, process_group, splits: List[int]):
        ctx.process_group = process_group
        ctx.dim = dim
        rank = dist.get_rank(process_group)
        ctx.grad_scale = splits[rank] / sum(splits)
        ctx.splits = splits
        return torch.split(input_, splits, dim=dim)[rank].clone()

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx.grad_scale
        grad_output = grad_output.contiguous()
        world_size = dist.get_world_size(ctx.process_group)
        shapes = [list(grad_output.shape) for _ in range(world_size)]
        for i, shape in enumerate(shapes):
            shape[ctx.dim] = ctx.splits[i]
        tensor_list = [torch.empty(shape, dtype=grad_output.dtype, device=grad_output.device) for shape in shapes]
        dist.all_gather(tensor_list, grad_output, group=ctx.process_group)
        return torch.cat(tensor_list, dim=ctx.dim), None, None, None


def split_forward_gather_backward_var_len(input_, dim, process_group, splits: List[int]):
    return _SplitForwardGatherBackwardVarLen.apply(input_, dim, process_group, splits)


class _GatherForwardSplitBackwardVarLen(torch.autograd.Function):
    """
    Split the input and keep only the corresponding chuck to the rank.

    Args:
        input_ (`torch.Tensor`): input matrix.
        dim (int): the dimension to perform split and gather
        process_group (`torch.distributed.ProcessGroup`): the process group used for collective communication

    """

    @staticmethod
    def forward(ctx, input_, dim, process_group, splits: List[int]):
        input_ = input_.contiguous()
        ctx.process_group = process_group
        ctx.dim = dim
        rank = dist.get_rank(process_group)

        ctx.grad_scale = sum(splits) / splits[rank]
        ctx.splits = splits
        world_size = dist.get_world_size(ctx.process_group)
        shapes = [list(input_.shape) for _ in range(world_size)]
        for i, shape in enumerate(shapes):
            shape[dim] = splits[i]
        tensor_list = [torch.empty(shape, dtype=input_.dtype, device=input_.device) for shape in shapes]
        dist.all_gather(tensor_list, input_, group=ctx.process_group)
        return torch.cat(tensor_list, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx.grad_scale
        rank = dist.get_rank(ctx.process_group)
        return torch.split(grad_output, ctx.splits, dim=ctx.dim)[rank].clone(), None, None, None


def gather_forward_split_backward_var_len(input_, dim, process_group, splits: List[int]):
    return _GatherForwardSplitBackwardVarLen.apply(input_, dim, process_group, splits)


def _fa_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float = 0.0, softmax_scale: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if SUPPORT_FA3:
        out, softmax_lse, *_ = _flash_attn_forward_v3(
            q,
            k,
            v,
            None,
            None,
            None,
            None,  # k_new, q_new, qv, out
            None,
            None,
            None,  # cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new
            None,
            None,
            None,
            None,  # seqused_q, seqused_k, max_seqlen_q, max_seqlen_k
            None,
            None,
            None,  # page_table, kv_batch_idx, leftpad_k
            None,
            None,  # rotary_cos/sin
            None,
            None,
            None,  # q_descale, k_descale, v_descale
            softmax_scale,
            False,  # causal
            (-1, -1),
        )
        rng_state = None
    else:
        out, softmax_lse, _, rng_state = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
    return out, softmax_lse, rng_state


def _fa_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    rng_state: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    deterministic: bool = False,
) -> None:
    if SUPPORT_FA3:
        _flash_attn_backward_v3(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None, None, None, None, None, None,
            dq,
            dk,
            dv,
            softmax_scale,
            False,  # causal
            (-1, -1),
            deterministic=deterministic,
        )
    else:
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            deterministic=deterministic,
            rng_state=rng_state,
        )


class RingAttention(torch.autograd.Function):
    ATTN_DONE: torch.cuda.Event = None
    SP_STREAM: torch.cuda.Stream = None

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sp_group: dist.ProcessGroup,
        sp_stream: torch.cuda.Stream,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        deterministic: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ring attention forward

        Args:
            ctx (_type_): self
            q (torch.Tensor): shape [B, S, N, D]
            k (torch.Tensor): shape [B, S, N, D]
            v (torch.Tensor): shape [B, S, N, D]
            sp_group (dist.ProcessGroup): sequence parallel group
            sp_stream (torch.cuda.Stream): sequence parallel stream
            dropout_p (float, optional): dropout prob. Defaults to 0.0.
            softmax_scale (Optional[float], optional): softmax scale. Defaults to None.
            deterministic (Optional[bool], optional): backward deterministic mode. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: output and log sum exp. Output's shape should be [B, S, N, D]. LSE's shape should be [B, N, S].
        """
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        sp_size = dist.get_world_size(sp_group)
        kv_comms: List[RingComm] = [RingComm(sp_group) for _ in range(2)]

        # [B, S, N, D]
        q, k, v = [x.contiguous() for x in [q, k, v]]
        # Pre-allocate double buffer for overlapping and receiving next step's inputs
        kv_buffers = [torch.stack((k, v))]  # (2, B, S, N, D)
        kv_buffers.append(torch.empty_like(kv_buffers[0]))
        # outputs
        out = None
        block_out = [None, None]
        softmax_lse = [None, None]
        block_softmax_lse = [None, None]  # log sum exp, the denominator of softmax in attention
        rng_states = [None for _ in range(sp_size)]
        sp_streams = [torch.cuda.current_stream(), sp_stream]

        def _kv_comm(i):
            # Avoid overwriting attn input when it shares mem with buffer
            if not RingAttention.ATTN_DONE.query():
                kv_buffers[(i + 1) % 2] = torch.empty_like(kv_buffers[i % 2])
            if i < sp_size - 1:
                kv_comms[i % 2].send_recv(kv_buffers[i % 2], kv_buffers[(i + 1) % 2])

        for i in range(sp_size):
            with torch.cuda.stream(sp_streams[i % 2]):
                # Wait for current kv from prev rank
                # NOTE: waiting outside the current stream will NOT correctly synchronize.
                if i == 0:
                    _kv_comm(i)
                else:
                    kv_comms[(i + 1) % 2].wait()
                kv_block = kv_buffers[i % 2]
                q_block = q
                block_out[i % 2], block_softmax_lse[i % 2], rng_states[i] = _fa_forward(
                    q_block, kv_block[0], kv_block[1], dropout_p, softmax_scale
                )
                RingAttention.ATTN_DONE.record()
                # Pipeline the next KV comm with output correction instead of the next flash attn
                # to minimize idle time when comm takes longer than attn.
                _kv_comm(i + 1)
                block_softmax_lse[i % 2] = (
                    block_softmax_lse[i % 2].transpose(1, 2).unsqueeze(-1).contiguous().float()
                )  # [B, N, S] -> [B, S, N, 1]
                assert block_out[i % 2].shape[:-1] == block_softmax_lse[i % 2].shape[:-1]
                # Output and log sum exp correction. Ideally overlap this with the next flash attn kernel.
                # In reality this always finishes before next flash attn; no need for extra sync.
                if i == 0:
                    out = block_out[0]
                    softmax_lse = block_softmax_lse[0]
                else:
                    out, softmax_lse = _rescale_out_lse(out, block_out[i % 2], softmax_lse, block_softmax_lse[i % 2])
        torch.cuda.current_stream().wait_stream(sp_stream)
        out = out.to(q.dtype)
        softmax_lse = softmax_lse.squeeze(-1).transpose(1, 2).contiguous()

        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.deterministic = deterministic
        ctx.sp_group = sp_group
        ctx.save_for_backward(q, k, v, out, softmax_lse, *rng_states)  # lse [B, N, S]
        return out, softmax_lse

    @staticmethod
    def backward(ctx, grad_output, grad_softmax_lse):
        # q, k, v, out: [B, S, N, D], softmax_lse: [B, N, S]
        q, k, v, out, softmax_lse, *rng_states = ctx.saved_tensors

        sp_group = ctx.sp_group
        sp_size = dist.get_world_size(sp_group)
        kv_comm = RingComm(sp_group)
        dkv_comm = RingComm(sp_group)

        grad_output = grad_output.contiguous()
        kv_buffers = [torch.stack((k, v))]  # (2, B, S, N, D)
        kv_buffers.append(torch.empty_like(kv_buffers[0]))
        dq = None
        dq_block = torch.empty_like(q)
        dk_block = torch.empty_like(k)
        dv_block = torch.empty_like(v)
        dkv_buffers = [torch.empty_like(kv, dtype=torch.float) for kv in kv_buffers]
        del k, v

        for i in range(sp_size):
            if i > 0:
                kv_comm.wait()
            if i < sp_size - 1:
                kv_comm.send_recv(kv_buffers[i % 2], kv_buffers[(i + 1) % 2])

            k_block, v_block = kv_buffers[i % 2]
            _fa_backward(
                grad_output,
                q,
                k_block,
                v_block,
                out,
                softmax_lse,
                dq_block,
                dk_block,
                dv_block,
                rng_states[i],
                dropout_p=ctx.dropout_p,
                softmax_scale=ctx.softmax_scale,
                deterministic=ctx.deterministic,
            )

            if i == 0:
                dq = dq_block.float()
                dkv_buffers[i % 2][0] = dk_block.float()
                dkv_buffers[i % 2][1] = dv_block.float()
            else:
                dq += dq_block
                dkv_comm.wait()
                dkv_buffers[i % 2][0] += dk_block
                dkv_buffers[i % 2][1] += dv_block
            dkv_comm.send_recv(dkv_buffers[i % 2], dkv_buffers[(i + 1) % 2])
        dkv_comm.wait()
        dkv = dkv_buffers[sp_size % 2]

        dq, dk, dv = [x.to(q.dtype) for x in (dq, *dkv)]

        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def attention(
        q,
        k,
        v,
        sp_group,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        deterministic: bool = False,
        return_softmax: bool = False,
    ):
        """Ring attention

        Args:
            q (torch.Tensor): shape [B, S, N, D]
            k (torch.Tensor): shape [B, S, N, D]
            v (torch.Tensor): shape [B, S, N, D]
            sp_group (dist.ProcessGroup): sequence parallel group
            dropout_p (float, optional): dropout prob. Defaults to 0.0.
            softmax_scale (Optional[float], optional): softmax scale. Defaults to None.
            deterministic (Optional[bool], optional): backward deterministic mode. Defaults to False.
            return_softmax (bool, optional): return softmax or not. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: output and log sum exp. Output's shape should be [B, S, N, D]. LSE's shape should be [B, N, S].
        """
        if RingAttention.ATTN_DONE is None:
            RingAttention.ATTN_DONE = torch.cuda.Event()
        if RingAttention.SP_STREAM is None:
            RingAttention.SP_STREAM = torch.cuda.Stream()
        out, softmax_lse = RingAttention.apply(
            q, k, v, sp_group, RingAttention.SP_STREAM, dropout_p, softmax_scale, deterministic
        )
        if return_softmax:
            return out, softmax_lse
        return out


def ring_attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, sp_group: dist.ProcessGroup) -> Tensor:
    if isinstance(pe, torch.Tensor):
        q, k = apply_rope(q, k, pe)
    else:
        cos, sin = pe
        q, k = LigerRopeFunction.apply(q, k, cos, sin)
    q, k, v = [x.transpose(1, 2) for x in (q, k, v)]  # [B, H, L, D] -> [B, L, H, D]
    x = RingAttention.attention(q, k, v, sp_group)
    x = rearrange(x, "B L H D -> B L (H D)")
    return x


class DistributedDoubleStreamBlockProcessor:
    def __init__(self, shard_config: ShardConfig) -> None:
        self.shard_config = shard_config

    def __call__(
        self, attn: DoubleStreamBlock, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor
    ) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        if attn.img_attn.fused_qkv:
            img_qkv = attn.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        else:
            img_q = rearrange(attn.img_attn.q_proj(img_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            img_k = rearrange(attn.img_attn.k_proj(img_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            img_v = rearrange(attn.img_attn.v_proj(img_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)
        if not attn.img_attn.fused_qkv:
            img_q = rearrange(img_q, "B L H D -> B H L D")
            img_k = rearrange(img_k, "B L H D -> B H L D")
            img_v = rearrange(img_v, "B L H D -> B H L D")

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        if attn.txt_attn.fused_qkv:
            txt_qkv = attn.txt_attn.qkv(txt_modulated)
            txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        else:
            txt_q = rearrange(attn.txt_attn.q_proj(txt_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            txt_k = rearrange(attn.txt_attn.k_proj(txt_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            txt_v = rearrange(attn.txt_attn.v_proj(txt_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)
        if not attn.txt_attn.fused_qkv:
            txt_q = rearrange(txt_q, "B L H D -> B H L D")
            txt_k = rearrange(txt_k, "B L H D -> B H L D")
            txt_v = rearrange(txt_v, "B L H D -> B H L D")

        txt_len = txt_q.size(2)
        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        if (
            self.shard_config.enable_sequence_parallelism
            and self.shard_config.sequence_parallelism_mode == "all_to_all"
        ):
            assert (
                attn.num_heads % self.shard_config.sequence_parallel_size == 0
            ), f"Expected num heads({attn.num_heads}) % sp size({self.shard_config.sequence_parallel_size}) == 0"
            # TODO: overlap the communication with computation
            q = all_to_all_comm(q, self.shard_config.sequence_parallel_process_group, scatter_dim=1, gather_dim=2)
            k = all_to_all_comm(k, self.shard_config.sequence_parallel_process_group, scatter_dim=1, gather_dim=2)
            v = all_to_all_comm(v, self.shard_config.sequence_parallel_process_group, scatter_dim=1, gather_dim=2)

        if self.shard_config.enable_sequence_parallelism and self.shard_config.sequence_parallelism_mode == "ring_attn":
            attn1 = ring_attention(q, k, v, pe, self.shard_config.sequence_parallel_process_group)
        else:
            attn1 = attention(q, k, v, pe=pe)
        if (
            self.shard_config.enable_sequence_parallelism
            and self.shard_config.sequence_parallelism_mode == "all_to_all"
        ):
            attn1 = all_to_all_comm(
                attn1, self.shard_config.sequence_parallel_process_group, scatter_dim=1, gather_dim=2
            )
        txt_attn, img_attn = attn1[:, :txt_len], attn1[:, txt_len:]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class DistributedSingleStreamBlockProcessor:
    def __init__(self, shard_config: ShardConfig) -> None:
        self.shard_config = shard_config

    def __call__(self, attn: SingleStreamBlock, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift

        if attn.fused_qkv:
            qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        else:
            q = rearrange(attn.q_proj(x_mod), "B L (H D) -> B L H D", H=attn.num_heads)
            k = rearrange(attn.k_proj(x_mod), "B L (H D) -> B L H D", H=attn.num_heads)
            v, mlp = torch.split(attn.v_mlp(x_mod), [attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
            v = rearrange(v, "B L (H D) -> B L H D", H=attn.num_heads)
        q, k = attn.norm(q, k, v)
        if not attn.fused_qkv:
            q = rearrange(q, "B L H D -> B H L D")
            k = rearrange(k, "B L H D -> B H L D")
            v = rearrange(v, "B L H D -> B H L D")

        if (
            self.shard_config.enable_sequence_parallelism
            and self.shard_config.sequence_parallelism_mode == "all_to_all"
        ):
            assert (
                attn.num_heads % self.shard_config.sequence_parallel_size == 0
            ), f"Expected num heads({attn.num_heads}) % sp size({self.shard_config.sequence_parallel_size}) == 0"
            q = all_to_all_comm(q, self.shard_config.sequence_parallel_process_group, scatter_dim=1, gather_dim=2)
            k = all_to_all_comm(k, self.shard_config.sequence_parallel_process_group, scatter_dim=1, gather_dim=2)
            v = all_to_all_comm(v, self.shard_config.sequence_parallel_process_group, scatter_dim=1, gather_dim=2)

        # compute attention
        if self.shard_config.enable_sequence_parallelism and self.shard_config.sequence_parallelism_mode == "ring_attn":
            attn_1 = ring_attention(q, k, v, pe, self.shard_config.sequence_parallel_process_group)
        else:
            attn_1 = attention(q, k, v, pe=pe)

        if (
            self.shard_config.enable_sequence_parallelism
            and self.shard_config.sequence_parallelism_mode == "all_to_all"
        ):
            attn_1 = all_to_all_comm(
                attn_1, self.shard_config.sequence_parallel_process_group, scatter_dim=1, gather_dim=2
            )

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = x + mod.gate * output
        return output


class _TempSwitchCP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, shard_config: ShardConfig, value: bool):
        ctx.old_value = shard_config.enable_sequence_parallelism
        ctx.shard_config = shard_config
        shard_config.enable_sequence_parallelism = value
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        print(f"in backward, sp mode: {ctx.shard_config.enable_sequence_parallelism}")
        ctx.shard_config.enable_sequence_parallelism = ctx.old_value
        return grad_output, None, None


def switch_sequence_parallelism(input_, shard_config: ShardConfig, value: bool):
    return _TempSwitchCP.apply(input_, shard_config, value)


def mmdit_model_forward(
    self: MMDiTModel,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y_vec: Tensor,
    cond: Tensor = None,
    guidance: Tensor | None = None,
    shard_config: ShardConfig = None,
    stage_index: Optional[List[int]] = None,
    internal_img: Optional[Tensor] = None,
    internal_txt: Optional[Tensor] = None,
    internal_pe: Optional[Tensor] = None,
    internal_vec: Optional[Tensor] = None,
    **kwargs,
):
    txt_len = txt.shape[1]
    if shard_config.pipeline_stage_manager is None or shard_config.pipeline_stage_manager.is_first_stage():
        img, txt, vec, pe = self.prepare_block_inputs(img, img_ids, txt, txt_ids, timesteps, y_vec, cond, guidance)
        has_grad = img.grad_fn is not None
        old_sequence_parallelism = shard_config.enable_sequence_parallelism
        if shard_config.enable_sequence_parallelism:
            assert (
                txt.shape[1] + img.shape[1]
            ) % shard_config.sequence_parallel_size == 0, (
                f"Expected {txt.shape[1] +img.shape[1]} % {shard_config.sequence_parallel_size} == 0"
            )
            mask = torch.zeros(txt.shape[1] + img.shape[1], dtype=bool)
            mask[txt.shape[1] :] = 1
            mask_chunks = mask.chunk(shard_config.sequence_parallel_size)
            cur_mask = mask_chunks[dist.get_rank(shard_config.sequence_parallel_process_group)]
            txt_splits = [len(c) - c.sum().item() for c in mask_chunks]
            img_splits = [c.sum().item() for c in mask_chunks]
            if 0 in img_splits:
                # temporarily disable sequence parallelism to avoid stucking
                img = switch_sequence_parallelism(img, shard_config, False)
            else:
                img = split_forward_gather_backward_var_len(
                    img, 1, shard_config.sequence_parallel_process_group, img_splits
                )
                txt = split_forward_gather_backward_var_len(
                    txt, 1, shard_config.sequence_parallel_process_group, txt_splits
                )
                if shard_config.sequence_parallelism_mode == "ring_attn":
                    # pe does not require grad
                    sp_rank = dist.get_rank(shard_config.sequence_parallel_process_group)
                    if isinstance(pe, torch.Tensor):
                        pe = pe.chunk(shard_config.sequence_parallel_size, dim=2)[sp_rank].clone()
                    else:
                        cos, sin = pe
                        cos = cos.chunk(shard_config.sequence_parallel_size, dim=1)[sp_rank].clone()
                        sin = sin.chunk(shard_config.sequence_parallel_size, dim=1)[sp_rank].clone()
                        pe = (cos, sin)
    else:
        img, txt, vec, pe = internal_img, internal_txt, internal_vec, internal_pe

    double_start, double_end = 0, len(self.double_blocks)
    if shard_config.pipeline_stage_manager is not None:
        double_start = stage_index[0]
        double_end = min(stage_index[1], len(self.double_blocks))

    for block in self.double_blocks[double_start:double_end]:
        img, txt = auto_grad_checkpoint(block, img, txt, vec, pe)

    if shard_config.pipeline_stage_manager is not None and stage_index[1] <= len(self.double_blocks):
        return {
            "internal_img": img,
            "internal_txt": txt,
            "internal_pe": pe,
            "internal_vec": vec,
        }
    single_start, single_end = 0, len(self.single_blocks)
    if shard_config.pipeline_stage_manager is not None:
        single_start = max(stage_index[0] - len(self.double_blocks), 0)
        single_end = stage_index[1] - len(self.double_blocks)

    if single_start == 0:
        img = torch.cat((txt, img), 1)

    for block in self.single_blocks[single_start:single_end]:
        img = auto_grad_checkpoint(block, img, vec, pe)

    if shard_config.pipeline_stage_manager is not None and single_end < len(self.single_blocks):
        return {
            "internal_img": img,
            "internal_pe": pe,
            "internal_vec": vec,
        }

    if shard_config.enable_sequence_parallelism:
        img = img[:, cur_mask]
    else:
        img = img[:, txt_len:]

    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

    if shard_config.enable_sequence_parallelism:
        img = gather_forward_split_backward_var_len(img, 1, shard_config.sequence_parallel_process_group, img_splits)

    if not has_grad:
        shard_config.enable_sequence_parallelism = old_sequence_parallelism
    return img


class MMDiTPolicy(Policy):
    def config_sanity_check(self):
        if self.shard_config.enable_sequence_parallelism and is_share_sp_tp(
            self.shard_config.sequence_parallelism_mode
        ):
            assert self.shard_config.enable_tensor_parallelism, "Tensor parallelism should be enabled"

    def preprocess(self) -> nn.Module:
        return self.model

    def postprocess(self) -> nn.Module:
        return self.model

    def tie_weight_check(self) -> bool:
        return False

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        policy = {
            DoubleStreamBlock: ModulePolicyDescription(attribute_replacement={}, sub_module_replacement=[]),
            SingleStreamBlock: ModulePolicyDescription(attribute_replacement={}, sub_module_replacement=[]),
        }

        if self.shard_config.enable_sequence_parallelism:
            if not is_share_sp_tp(self.shard_config.sequence_parallelism_mode):
                policy[DoubleStreamBlock].attribute_replacement["processor"] = DistributedDoubleStreamBlockProcessor(
                    self.shard_config
                )
                policy[SingleStreamBlock].attribute_replacement["processor"] = DistributedSingleStreamBlockProcessor(
                    self.shard_config
                )
        if self.shard_config.enable_sequence_parallelism or self.shard_config.pipeline_stage_manager is not None:
            fwd_fn = partial(mmdit_model_forward, shard_config=self.shard_config)
            if self.shard_config.pipeline_stage_manager is not None:
                layers_per_stage = self.shard_config.pipeline_stage_manager.distribute_layers(
                    len(self.model.double_blocks) + len(self.model.single_blocks)
                )
                if self.shard_config.pipeline_stage_manager.is_interleave:
                    self.shard_config.pipeline_stage_manager.stage_indices = (
                        self.shard_config.pipeline_stage_manager.get_stage_index(layers_per_stage)
                    )
                else:
                    stage_index = self.shard_config.pipeline_stage_manager.get_stage_index(layers_per_stage)
                    fwd_fn = partial(mmdit_model_forward, shard_config=self.shard_config, stage_index=stage_index)
            self.append_or_create_method_replacement(
                description={
                    "forward": fwd_fn,
                },
                policy=policy,
                target_key=MMDiTModel,
            )

        if self.shard_config.enable_tensor_parallelism:
            mlp_hidden_size = int(self.model.config.hidden_size * self.model.config.mlp_ratio)
            assert (
                self.model.config.num_heads % self.shard_config.tensor_parallel_size == 0
                and mlp_hidden_size % self.shard_config.tensor_parallel_size == 0
            ), "num_heads and hidden_size should be divisible by tensor_parallel_size"
            for n in ["img", "txt"]:
                if self.model.config.fused_qkv:
                    policy[DoubleStreamBlock].sub_module_replacement.append(
                        SubModuleReplacementDescription(
                            suffix=f"{n}_attn.qkv",
                            target_module=FusedLinear1D_Col,
                            kwargs={
                                "split_sizes": [self.model.config.hidden_size] * 3,
                                "seq_parallel_mode": self.shard_config.sequence_parallelism_mode,
                            },
                        ),
                    )
                else:
                    policy[DoubleStreamBlock].sub_module_replacement.extend(
                        [
                            SubModuleReplacementDescription(
                                suffix=f"{n}_attn.q_proj",
                                target_module=Linear1D_Col,
                                kwargs={"seq_parallel_mode": self.shard_config.sequence_parallelism_mode},
                            ),
                            SubModuleReplacementDescription(
                                suffix=f"{n}_attn.k_proj",
                                target_module=Linear1D_Col,
                                kwargs={"seq_parallel_mode": self.shard_config.sequence_parallelism_mode},
                            ),
                            SubModuleReplacementDescription(
                                suffix=f"{n}_attn.v_proj",
                                target_module=Linear1D_Col,
                                kwargs={"seq_parallel_mode": self.shard_config.sequence_parallelism_mode},
                            ),
                        ]
                    )
                policy[DoubleStreamBlock].sub_module_replacement.extend(
                    [
                        SubModuleReplacementDescription(
                            suffix=f"{n}_attn.proj",
                            target_module=Linear1D_Row,
                            kwargs={"seq_parallel_mode": self.shard_config.sequence_parallelism_mode},
                        ),
                        SubModuleReplacementDescription(
                            suffix=f"{n}_mlp[0]",
                            target_module=Linear1D_Col,
                            kwargs={"seq_parallel_mode": self.shard_config.sequence_parallelism_mode},
                        ),
                        SubModuleReplacementDescription(
                            suffix=f"{n}_mlp[2]",
                            target_module=Linear1D_Row,
                            kwargs={"seq_parallel_mode": self.shard_config.sequence_parallelism_mode},
                        ),
                    ]
                )
            policy[DoubleStreamBlock].attribute_replacement["num_heads"] = (
                self.model.config.num_heads // self.shard_config.tensor_parallel_size
            )
            policy[SingleStreamBlock].attribute_replacement.update(
                {
                    "num_heads": self.model.config.num_heads // self.shard_config.tensor_parallel_size,
                    "hidden_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                    "mlp_hidden_dim": mlp_hidden_size // self.shard_config.tensor_parallel_size,
                }
            )
            if self.model.config.fused_qkv:
                policy[SingleStreamBlock].sub_module_replacement.append(
                    SubModuleReplacementDescription(
                        suffix="linear1",
                        target_module=FusedLinear1D_Col,
                        kwargs={
                            "split_sizes": [self.model.config.hidden_size] * 3 + [mlp_hidden_size],
                            "seq_parallel_mode": self.shard_config.sequence_parallelism_mode,
                        },
                    ),
                )
            else:
                policy[SingleStreamBlock].sub_module_replacement.extend(
                    [
                        SubModuleReplacementDescription(
                            suffix="q_proj",
                            target_module=Linear1D_Col,
                            kwargs={"seq_parallel_mode": self.shard_config.sequence_parallelism_mode},
                        ),
                        SubModuleReplacementDescription(
                            suffix="k_proj",
                            target_module=Linear1D_Col,
                            kwargs={"seq_parallel_mode": self.shard_config.sequence_parallelism_mode},
                        ),
                        SubModuleReplacementDescription(
                            suffix="v_mlp",
                            target_module=FusedLinear1D_Col,
                            kwargs={
                                "split_sizes": [self.model.config.hidden_size] + [mlp_hidden_size],
                                "seq_parallel_mode": self.shard_config.sequence_parallelism_mode,
                            },
                        ),
                    ]
                )
            policy[SingleStreamBlock].sub_module_replacement.extend(
                [
                    SubModuleReplacementDescription(
                        suffix="linear2",
                        target_module=FusedLinear1D_Row,
                        kwargs={
                            "split_sizes": [self.model.config.hidden_size, mlp_hidden_size],
                            "seq_parallel_mode": self.shard_config.sequence_parallelism_mode,
                        },
                    ),
                ],
            )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        stage_manager = self.shard_config.pipeline_stage_manager
        assert stage_manager is not None, "Pipeline stage manager is not set"

        held_layers = []
        total_blocks = [*self.model.double_blocks, *self.model.single_blocks]
        if stage_manager.is_first_stage(ignore_chunk=stage_manager.is_interleave):
            held_layers.extend(
                [
                    self.model.pe_embedder,
                    self.model.img_in,
                    self.model.time_in,
                    self.model.vector_in,
                    self.model.guidance_in,
                    self.model.cond_in,
                    self.model.txt_in,
                ]
            )

        layers_per_stage = stage_manager.distribute_layers(len(total_blocks))
        if stage_manager.is_interleave:
            assert stage_manager.num_model_chunks is not None
            stage_indices = stage_manager.get_stage_index(layers_per_stage)
            for start_idx, end_idx in stage_indices:
                held_layers.extend(total_blocks[start_idx:end_idx])
        else:
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
            held_layers.extend(total_blocks[start_idx:end_idx])
        if stage_manager.is_last_stage(ignore_chunk=stage_manager.is_interleave):
            held_layers.append(self.model.final_layer)
        return held_layers
