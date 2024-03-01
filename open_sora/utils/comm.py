import torch
import torch.distributed as dist
from colossalai.moe._operation import MoeInGradScaler, MoeOutGradScaler
from colossalai.shardformer.layer._operation import gather_forward_split_backward


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
