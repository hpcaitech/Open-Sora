import torch
import torch.distributed as dist


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
