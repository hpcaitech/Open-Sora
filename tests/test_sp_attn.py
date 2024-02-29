import colossalai
import pytest
import torch
import torch.distributed as dist
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer.layer._operation import gather_forward_split_backward
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device

from open_sora.modeling.dit import CrossAttention, SeqParallelCrossAttention


def check_sp_attn():
    sp_size = dist.get_world_size()
    sp_rank = dist.get_rank()
    q_dim, context_dim = 8, 4
    num_heads = 4
    head_dim = 16
    bs = 2
    sq = 8
    skv = 4
    attn = CrossAttention(q_dim, context_dim, num_heads, head_dim).to(
        get_current_device()
    )
    parallel_attn = SeqParallelCrossAttention(
        q_dim, context_dim, num_heads, head_dim, seq_parallel_group=dist.group.WORLD
    ).to(get_current_device())
    parallel_attn.load_state_dict(attn.state_dict())
    hidden_states = torch.rand(bs, sq, q_dim, device=get_current_device())
    context = torch.rand(bs, skv, context_dim, device=get_current_device())
    mask = torch.zeros(bs, 1, sq, skv, device=get_current_device())
    target = attn(hidden_states, context, mask)
    output_parallel = parallel_attn(
        hidden_states.chunk(sp_size, dim=1)[sp_rank],
        context.chunk(sp_size, dim=1)[sp_rank],
        mask,
    )
    assert torch.equal(target.chunk(sp_size, dim=1)[sp_rank], output_parallel)
    output = gather_forward_split_backward(output_parallel, 1, dist.group.WORLD)
    assert torch.equal(target, output)


def run_dist(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(
        config={},
        rank=rank,
        world_size=world_size,
        port=port,
        host="localhost",
        backend="nccl",
    )
    check_sp_attn()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_seq_parallel_attn():
    spawn(run_dist, 2)


if __name__ == "__main__":
    test_seq_parallel_attn()
