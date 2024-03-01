from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from colossalai.moe._operation import MoeInGradScaler, MoeOutGradScaler
from colossalai.shardformer.layer._operation import gather_forward_split_backward
from torch.distributed.distributed_c10d import get_global_rank


def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [
        t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        return _all_to_all(
            input_, ctx.world_size, process_group, scatter_dim, gather_dim
        )

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _all_to_all(
                grad_output,
                ctx.world_size,
                ctx.process_group,
                ctx.gather_dim,
                ctx.scatter_dim,
            ),
            None,
            None,
            None,
        )


def all_to_all(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)


def split_seq(input_: torch.Tensor, sp_size: int, sp_rank: int, dim: int = 1):
    """Split a tensor along sequence dimension. It will split input and divide grad by sp_size.

    Args:
        input_ (torch.Tensor): The common shape is (bs, seq, *).
        sp_size (int): Sequence parallel size.
        sp_rank (int): Sequence parallel rank.
        dim (int, optional): Sequence dimension. Defaults to 1.
    """
    input_ = input_.chunk(sp_size, dim=dim)[sp_rank].clone()
    return MoeOutGradScaler.apply(input_, sp_size)


def gather_seq(
    input_: torch.Tensor,
    sp_size: int,
    sp_rank: int,
    sp_group: dist.ProcessGroup,
    dim: int = 1,
):
    """Gather a tensor along sequence dimension. It will gather input and multiply grad by sp_size.

    Args:
        input_ (torch.Tensor): The common shape is (bs, seq, *).
        sp_size (int): Sequence parallel size.
        sp_rank (int): Sequence parallel rank.
        dim (int, optional): Sequence dimension. Defaults to 1.
    """
    input_ = gather_forward_split_backward(input_, dim, sp_group)
    return MoeInGradScaler.apply(input_, sp_size)


class AsyncAllGatherProjForTwo(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        q_proj_weight: torch.Tensor,
        q_proj_bias: Optional[torch.Tensor],
        k_proj_weight: torch.Tensor,
        k_proj_bias: Optional[torch.Tensor],
        v_proj_weight: torch.Tensor,
        v_proj_bias: Optional[torch.Tensor],
        dim: int,
        process_group: dist.ProcessGroup,
        sp_size: int,
        sp_rank: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert sp_size == 2
        ctx.process_group = process_group
        ctx.sp_size = sp_size
        ctx.sp_rank = sp_rank
        ctx.dim = dim

        is_cross_attn = not (hidden_states is context)
        ctx.is_cross_attn = is_cross_attn

        recv_hidden_states = torch.empty_like(hidden_states)
        if is_cross_attn:
            recv_context = torch.empty_like(context)
        else:
            recv_context = recv_hidden_states

        peer_global_rank = get_global_rank(process_group, 1 - sp_rank)
        ops = [
            dist.P2POp(dist.isend, hidden_states, peer_global_rank, process_group),
            dist.P2POp(dist.irecv, recv_hidden_states, peer_global_rank, process_group),
        ]
        if is_cross_attn:
            ops.extend(
                [
                    dist.P2POp(dist.isend, context, peer_global_rank, process_group),
                    dist.P2POp(
                        dist.irecv, recv_context, peer_global_rank, process_group
                    ),
                ]
            )

        reqs = dist.batch_isend_irecv(ops)

        # [B, S/P, H]
        q = F.linear(hidden_states, q_proj_weight, q_proj_bias)
        k = F.linear(context, k_proj_weight, k_proj_bias)
        v = F.linear(context, v_proj_weight, v_proj_bias)

        for req in reqs:
            req.wait()

        q_other = F.linear(recv_hidden_states, q_proj_weight, q_proj_bias)
        k_other = F.linear(recv_context, k_proj_weight, k_proj_bias)
        v_other = F.linear(recv_context, v_proj_weight, v_proj_bias)

        if sp_rank == 0:
            q = torch.cat([q, q_other], dim=dim)
            k = torch.cat([k, k_other], dim=dim)
            v = torch.cat([v, v_other], dim=dim)
        else:
            q = torch.cat([q_other, q], dim=dim)
            k = torch.cat([k_other, k], dim=dim)
            v = torch.cat([v_other, v], dim=dim)

        ctx.save_for_backward(
            hidden_states,
            context,
            q_proj_weight,
            q_proj_bias,
            k_proj_weight,
            k_proj_bias,
            v_proj_weight,
            v_proj_bias,
            recv_hidden_states,
            recv_context,
        )

        return q, k, v

    @staticmethod
    def backward(
        ctx: torch.Any, q_grad: torch.Tensor, k_grad: torch.Tensor, v_grad: torch.Tensor
    ) -> torch.Any:
        (
            hidden_states,
            context,
            q_proj_weight,
            q_proj_bias,
            k_proj_weight,
            k_proj_bias,
            v_proj_weight,
            v_proj_bias,
            recv_hidden_states,
            recv_context,
        ) = ctx.saved_tensors

        # compute param grads
        if ctx.sp_rank == 0:
            hidden_states = torch.cat([hidden_states, recv_hidden_states], dim=ctx.dim)
            if ctx.is_cross_attn:
                context = torch.cat([context, recv_context], dim=ctx.dim)
            else:
                context = hidden_states
        else:
            hidden_states = torch.cat([recv_hidden_states, hidden_states], dim=ctx.dim)
            if ctx.is_cross_attn:
                context = torch.cat([recv_context, context], dim=ctx.dim)
            else:
                context = hidden_states
        q_proj_weight_grad = q_grad.transpose(-1, -2).matmul(hidden_states).sum(dim=0)
        q_proj_bias_grad = (
            q_grad.sum(dim=0).sum(dim=0) if q_proj_bias is not None else None
        )
        k_proj_weight_grad = k_grad.transpose(-1, -2).matmul(context).sum(dim=0)
        k_proj_bias_grad = (
            k_grad.sum(dim=0).sum(dim=0) if k_proj_bias is not None else None
        )
        v_proj_weight_grad = v_grad.transpose(-1, -2).matmul(context).sum(dim=0)
        v_proj_bias_grad = (
            v_grad.sum(dim=0).sum(dim=0) if v_proj_bias is not None else None
        )

        # split grads
        q_grad = q_grad.chunk(ctx.sp_size, dim=ctx.dim)[ctx.sp_rank].clone()
        k_grad = k_grad.chunk(ctx.sp_size, dim=ctx.dim)[ctx.sp_rank].clone()
        v_grad = v_grad.chunk(ctx.sp_size, dim=ctx.dim)[ctx.sp_rank].clone()

        if ctx.is_cross_attn:
            hidden_states_grad = torch.matmul(q_grad, q_proj_weight)
            context_grad = torch.matmul(k_grad, k_proj_weight) + torch.matmul(
                v_grad, v_proj_weight
            )
        else:
            hidden_states_grad = (
                torch.matmul(q_grad, q_proj_weight)
                + torch.matmul(k_grad, k_proj_weight)
                + torch.matmul(v_grad, v_proj_weight)
            )
            context_grad = hidden_states_grad

        return (
            hidden_states_grad,
            context_grad,
            q_proj_weight_grad,
            q_proj_bias_grad,
            k_proj_weight_grad,
            k_proj_bias_grad,
            v_proj_weight_grad,
            v_proj_bias_grad,
            None,
            None,
            None,
            None,
        )


def async_all_gather_proj_for_two(
    hidden_states: torch.Tensor,
    context: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: Optional[torch.Tensor],
    k_proj_weight: torch.Tensor,
    k_proj_bias: Optional[torch.Tensor],
    v_proj_weight: torch.Tensor,
    v_proj_bias: Optional[torch.Tensor],
    dim: int,
    process_group: dist.ProcessGroup,
    sp_size: int,
    sp_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return AsyncAllGatherProjForTwo.apply(
        hidden_states,
        context,
        q_proj_weight,
        q_proj_bias,
        k_proj_weight,
        k_proj_bias,
        v_proj_weight,
        v_proj_bias,
        dim,
        process_group,
        sp_size,
        sp_rank,
    )
