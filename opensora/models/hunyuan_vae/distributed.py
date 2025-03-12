from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from colossalai.shardformer.layer._operation import gather_forward_split_backward, split_forward_gather_backward
from colossalai.shardformer.layer.attn import RingComm, _rescale_out_lse
from colossalai.shardformer.layer.utils import SeqParallelUtils
from diffusers.models.attention_processor import Attention

from opensora.models.vae.tensor_parallel import Conv3dTPRow
from opensora.models.vae.utils import get_conv3d_n_chunks

from .unet_causal_3d_blocks import UpsampleCausal3D

try:
    from xformers.ops.fmha import (
        Context,
        Inputs,
        _memory_efficient_attention_backward,
        _memory_efficient_attention_forward_requires_grad,
    )

    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False

SEQ_ALIGN = 32
SEQ_LIMIT = 16 * 1024


def align_atten_bias(attn_bias):
    B, N, S, S = attn_bias.shape
    align_size = 8
    if S % align_size != 0:
        expand_S = (S // align_size + 1) * align_size
        new_shape = [B, N, S, expand_S]
        attn_bias = torch.empty(new_shape, dtype=attn_bias.dtype, device=attn_bias.device)[:, :, :, :S].copy_(attn_bias)
    return attn_bias


def _attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
):
    attn_bias = align_atten_bias(attn_bias)
    inp = Inputs(q, k, v, attn_bias, p=0, scale=scale, is_partial=False)
    out, ctx = _memory_efficient_attention_forward_requires_grad(inp, None)

    S = attn_bias.shape[-2]
    if ctx.lse.shape[-1] != S:
        ctx.lse = ctx.lse[:, :, :S]
    return out, ctx.lse, ctx.rng_state


def _attn_bwd(
    grad: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    rng_state: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
):
    attn_bias = align_atten_bias(attn_bias)
    inp = Inputs(q, k, v, attn_bias, p=0, scale=scale, output_dtype=q.dtype, is_partial=False)
    ctx = Context(lse, out, rng_state=rng_state)
    grads = _memory_efficient_attention_backward(ctx, inp, grad, None)
    return grads.dq, grads.dk, grads.dv


class MemEfficientRingAttention(torch.autograd.Function):
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
        softmax_scale: Optional[float] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ring attention forward

        Args:
            ctx (_type_): self
            q (torch.Tensor): shape [B, S/P, N, D]
            k (torch.Tensor): shape [B, S/P, N, D]
            v (torch.Tensor): shape [B, S/P, N, D]
            sp_group (dist.ProcessGroup): sequence parallel group
            sp_stream (torch.cuda.Stream): sequence parallel stream
            softmax_scale (Optional[float], optional): softmax scale. Defaults to None.
            attn_mask (Optional[torch.Tensor], optional): attention mask shape [B, N, S/P, S]. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: output and log sum exp. Output's shape should be [B, S/P, N, D]. LSE's shape should be [B, N, S/P].
        """
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)
        kv_comms: List[RingComm] = [RingComm(sp_group) for _ in range(2)]
        block_attn_masks = [None] * sp_size
        if attn_mask is not None:
            # if attn_mask is splitted, uncomment the following line
            # attn_mask = attn_mask.chunk(sp_size, dim=2)[sp_rank]
            block_attn_masks = attn_mask.chunk(sp_size, dim=-1)

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
            if not MemEfficientRingAttention.ATTN_DONE.query():
                kv_buffers[(i + 1) % 2] = torch.empty_like(kv_buffers[i % 2])
            if i < sp_size - 1:
                kv_comms[i % 2].send_recv(kv_buffers[i % 2], kv_buffers[(i + 1) % 2])

        block_idx = sp_rank
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
                block_out[i % 2], block_softmax_lse[i % 2], rng_states[i] = _attn_fwd(
                    q_block, kv_block[0], kv_block[1], attn_bias=block_attn_masks[block_idx], scale=softmax_scale
                )
                MemEfficientRingAttention.ATTN_DONE.record()
                # Pipeline the next KV comm with output correction instead of the next flash attn
                # to minimize idle time when comm takes longer than attn.
                _kv_comm(i + 1)
                block_softmax_lse[i % 2] = (
                    block_softmax_lse[i % 2].transpose(1, 2).unsqueeze(-1).contiguous().float()
                )  # [B, N, S] -> [B, S, N, 1]
                assert (
                    block_out[i % 2].shape[:-1] == block_softmax_lse[i % 2].shape[:-1]
                ), f"{block_out[i % 2].shape} != {block_softmax_lse[i % 2].shape}"
                # Output and log sum exp correction. Ideally overlap this with the next flash attn kernel.
                # In reality this always finishes before next flash attn; no need for extra sync.
                if i == 0:
                    out = block_out[0]
                    softmax_lse = block_softmax_lse[0]
                else:
                    out, softmax_lse = _rescale_out_lse(out, block_out[i % 2], softmax_lse, block_softmax_lse[i % 2])
                block_idx = (block_idx - 1) % sp_size
        torch.cuda.current_stream().wait_stream(sp_stream)
        out = out.to(q.dtype)
        softmax_lse = softmax_lse.squeeze(-1).transpose(1, 2).contiguous()

        ctx.softmax_scale = softmax_scale
        ctx.block_attn_masks = block_attn_masks
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
        dkv_buffers = [torch.empty_like(kv, dtype=torch.float) for kv in kv_buffers]
        del k, v

        block_idx = dist.get_rank(sp_group)
        for i in range(sp_size):
            if i > 0:
                kv_comm.wait()
            if i < sp_size - 1:
                kv_comm.send_recv(kv_buffers[i % 2], kv_buffers[(i + 1) % 2])

            k_block, v_block = kv_buffers[i % 2]
            dq_block, dk_block, dv_block = _context_chunk_attn_bwd(
                grad_output,
                q,
                k_block,
                v_block,
                out,
                softmax_lse,
                rng_states[i],
                attn_bias=ctx.block_attn_masks[block_idx],
                scale=ctx.softmax_scale,
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
            block_idx = (block_idx - 1) % sp_size
        dkv_comm.wait()
        dkv = dkv_buffers[sp_size % 2]

        dq, dk, dv = [x.to(q.dtype) for x in (dq, *dkv)]

        torch.cuda.empty_cache()
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def attention(
        q,
        k,
        v,
        sp_group,
        softmax_scale: Optional[float] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_softmax: bool = False,
    ):
        """Ring attention

        Args:
            q (torch.Tensor): shape [B, S, N, D]
            k (torch.Tensor): shape [B, S, N, D]
            v (torch.Tensor): shape [B, S, N, D]
            sp_group (dist.ProcessGroup): sequence parallel group
            softmax_scale (Optional[float], optional): softmax scale. Defaults to None.
            attn_mask (Optional[torch.Tensor], optional): attention mask. Defaults to None.
            return_softmax (bool, optional): return softmax or not. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: output and log sum exp. Output's shape should be [B, S, N, D]. LSE's shape should be [B, N, S].
        """
        if MemEfficientRingAttention.ATTN_DONE is None:
            MemEfficientRingAttention.ATTN_DONE = torch.cuda.Event()
        if MemEfficientRingAttention.SP_STREAM is None:
            MemEfficientRingAttention.SP_STREAM = torch.cuda.Stream()
        out, softmax_lse = MemEfficientRingAttention.apply(
            q, k, v, sp_group, MemEfficientRingAttention.SP_STREAM, softmax_scale, attn_mask
        )
        if return_softmax:
            return out, softmax_lse
        return out


class MemEfficientRingAttnProcessor:
    def __init__(self, sp_group: dist.ProcessGroup):
        self.sp_group = sp_group
        if not HAS_XFORMERS:
            raise ImportError("MemEfficientRingAttnProcessor requires xformers, to use it, please install xformers.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        sp_group = self.sp_group
        assert sp_group is not None, "sp_group must be provided for MemEfficientRingAttnProcessor"

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        hidden_states = split_forward_gather_backward(hidden_states, 1, sp_group)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim)

        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        assert (
            query.shape[1] % dist.get_world_size(sp_group) == 0
        ), f"sequence length ({query.shape[1]}) must be divisible by sp_group size ({dist.get_world_size(sp_group)})"

        hidden_states = MemEfficientRingAttention.attention(query, key, value, sp_group, attn_mask=attention_mask)

        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = gather_forward_split_backward(hidden_states, 1, sp_group)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class ContextParallelAttention:
    def __init__(self):
        raise ImportError(f"ContextParallelAttention should not be initialized directly.")

    @staticmethod
    def from_native_module(module: Attention, process_group, *args, **kwargs) -> Attention:
        """
        Convert a native RMSNorm module to colossalai layer norm module,
        and optionally mark parameters for gradient aggregation.

        Args:
            module (nn.Module): The native RMSNorm module to be converted.
            sp_partial_derived (bool): Whether this module's gradients are partially derived in sequence parallelism.

        Returns:
            nn.Module: The RMSNorm module.
        """

        # Since gradients are computed using only a subset of the data,
        # aggregation of these gradients is necessary during backpropagation.
        # Therefore, we annotate these parameters in advance to indicate the need for gradient aggregation.
        SeqParallelUtils.marked_as_sp_partial_derived_param(module.to_q.weight)
        SeqParallelUtils.marked_as_sp_partial_derived_param(module.to_k.weight)
        SeqParallelUtils.marked_as_sp_partial_derived_param(module.to_v.weight)

        if module.to_q.bias is not None:
            SeqParallelUtils.marked_as_sp_partial_derived_param(module.to_q.bias)
            SeqParallelUtils.marked_as_sp_partial_derived_param(module.to_k.bias)
            SeqParallelUtils.marked_as_sp_partial_derived_param(module.to_v.bias)

        module.set_processor(MemEfficientRingAttnProcessor(process_group))

        return module


def _context_chunk_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    scale: Optional[float],
    seq_align: int = SEQ_ALIGN,
    seq_limit: int = SEQ_LIMIT,
):
    seq_len = q.shape[1]
    n_chunks = get_conv3d_n_chunks(seq_len, seq_align, seq_limit)
    q_chunks, k_chunks, v_chunks = q.chunk(n_chunks, dim=1), k.chunk(n_chunks, dim=1), v.chunk(n_chunks, dim=1)
    attn_bias_chunks = attn_bias.chunk(n_chunks, dim=2) if attn_bias is not None else [None] * n_chunks
    out_chunks = []
    lse_chunks = []
    rng_states = []
    for q_chunk, attn_bias_chunk in zip(q_chunks, attn_bias_chunks):
        inner_attn_bias_chunks = (
            attn_bias_chunk.chunk(n_chunks, dim=3) if attn_bias_chunk is not None else [None] * n_chunks
        )
        out_chunk = None
        for k_chunk, v_chunk, inner_attn_bias_chunk in zip(k_chunks, v_chunks, inner_attn_bias_chunks):
            block_out, block_lse, rng_state = _attn_fwd(q_chunk, k_chunk, v_chunk, inner_attn_bias_chunk, scale)
            block_lse = block_lse.transpose(1, 2).unsqueeze(-1).contiguous().float()  # [B, N, S] -> [B, S, N, 1]
            rng_states.append(rng_state)
            if out_chunk is None:
                out_chunk = block_out
                lse_chunk = block_lse
            else:
                out_chunk, lse_chunk = _rescale_out_lse(out_chunk, block_out, lse_chunk, block_lse)
            lse_chunk = lse_chunk.squeeze(-1).transpose(1, 2).contiguous()  # [B, S, N, 1] -> [B, N, S]
        out_chunks.append(out_chunk)
        lse_chunks.append(lse_chunk)
    out = torch.cat(out_chunks, dim=1)
    lse = torch.cat(lse_chunks, dim=-1)
    return out, lse, rng_states


def _context_chunk_attn_bwd(
    grad: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    rng_states: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    seq_align: int = SEQ_ALIGN,
    seq_limit: int = SEQ_LIMIT,
    fast_accum: bool = False,
):
    seq_len = q.shape[1]
    n_chunks = get_conv3d_n_chunks(seq_len, seq_align, seq_limit)
    if n_chunks == 1:
        return _attn_bwd(grad, q, k, v, out, lse, rng_states, attn_bias, scale)

    q_chunks, k_chunks, v_chunks = q.chunk(n_chunks, dim=1), k.chunk(n_chunks, dim=1), v.chunk(n_chunks, dim=1)
    attn_bias_chunks = attn_bias.chunk(n_chunks, dim=2) if attn_bias is not None else [None] * n_chunks
    out_chunks = out.chunk(n_chunks, dim=1)
    dout_chunks = grad.chunk(n_chunks, dim=1)
    lse_chunks = lse.chunk(n_chunks, dim=-1)
    if rng_states is None:
        rng_states = [None] * (n_chunks * n_chunks)

    i = 0

    acc_dtype = q.dtype if fast_accum else torch.float

    dq = torch.zeros_like(q, dtype=acc_dtype)
    dk = torch.zeros_like(k, dtype=acc_dtype)
    dv = torch.zeros_like(v, dtype=acc_dtype)

    dq_chunks = dq.chunk(n_chunks, dim=1)
    dk_chunks = dk.chunk(n_chunks, dim=1)
    dv_chunks = dv.chunk(n_chunks, dim=1)

    for q_idx in range(n_chunks):
        q_chunk = q_chunks[q_idx]
        attn_bias_chunk = attn_bias_chunks[q_idx]
        inner_attn_bias_chunks = (
            attn_bias_chunk.chunk(n_chunks, dim=3) if attn_bias_chunk is not None else [None] * n_chunks
        )
        out_chunk = out_chunks[q_idx]
        dout_chunk = dout_chunks[q_idx]
        lse_chunk = lse_chunks[q_idx]
        dq_acc = dq_chunks[q_idx]

        for kv_idx in range(n_chunks):
            k_chunk = k_chunks[kv_idx]
            v_chunk = v_chunks[kv_idx]
            inner_attn_bias_chunk = inner_attn_bias_chunks[kv_idx]
            dk_acc = dk_chunks[kv_idx]
            dv_acc = dv_chunks[kv_idx]

            block_dq, block_dk, block_dv = _attn_bwd(
                dout_chunk, q_chunk, k_chunk, v_chunk, out_chunk, lse_chunk, rng_states[i], inner_attn_bias_chunk, scale
            )

            dq_acc += block_dq
            dk_acc += block_dk
            dv_acc += block_dv
            i += 1

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


def prepare_parallel_causal_attention_mask(
    parallel_rank: int, parallel_size: int, n_frame: int, n_hw: int, dtype, device, batch_size: int = None
):
    seq_len = n_frame * n_hw
    assert seq_len % parallel_size == 0, f"seq_len {seq_len} must be divisible by parallel_size {parallel_size}"
    local_seq_len = seq_len // parallel_size
    local_seq_start = local_seq_len * parallel_rank
    if dtype is torch.bfloat16:
        # A trick to avoid nan of memory efficient attention, maybe introduce some bias
        fmin = torch.finfo(torch.float16).min
    else:
        fmin = torch.finfo(dtype).min
    mask = torch.full((local_seq_len, seq_len), fmin, dtype=dtype, device=device)
    for i in range(local_seq_len):
        i_frame = (i + local_seq_start) // n_hw
        mask[i, : (i_frame + 1) * n_hw] = 0
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


def prepare_parallel_attention_mask(
    self, hidden_states: torch.Tensor, cp_group: dist.ProcessGroup = None
) -> torch.Tensor:
    B, C, T, H, W = hidden_states.shape
    attention_mask = prepare_parallel_causal_attention_mask(
        dist.get_rank(cp_group),
        dist.get_world_size(cp_group),
        T,
        H * W,
        hidden_states.dtype,
        hidden_states.device,
        batch_size=B,
    )
    return attention_mask


class TPUpDecoderBlockCausal3D(UpsampleCausal3D):
    def __init__(
        self,
        channels,
        out_channels=None,
        kernel_size=3,
        bias=True,
        upsample_factor=(2, 2, 2),
        tp_group=None,
        split_input: bool = False,
        split_output: bool = False,
        conv_=None,
        shortcut_=None,
    ):
        assert tp_group is not None, "tp_group must be provided"
        super().__init__(channels, out_channels, kernel_size, bias, upsample_factor)
        conv = conv_ if conv_ is not None else self.conv.conv
        self.conv.conv = Conv3dTPRow.from_native_module(
            conv, tp_group, split_input=split_input, split_output=split_output
        )
        self.tp_group = tp_group
        tp_size = dist.get_world_size(group=self.tp_group)
        assert self.channels % tp_size == 0, f"channels {self.channels} must be divisible by tp_size {tp_size}"
        self.channels = self.channels // tp_size

    def forward(self, input_tensor):
        input_tensor = split_forward_gather_backward(input_tensor, 1, self.tp_group)
        return super().forward(input_tensor)

    def from_native_module(module: UpsampleCausal3D, process_group, **kwargs):
        conv = module.conv.conv
        return TPUpDecoderBlockCausal3D(
            module.channels,
            module.out_channels,
            conv.kernel_size[0],
            conv.bias is not None,
            module.upsample_factor,
            conv_=conv,
            shortcut_=getattr(module, "shortcut", None),
            tp_group=process_group,
            **kwargs,
        )
