import colossalai
import pytest
import torch
import torch.distributed as dist
from colossalai.logging import disable_existing_loggers
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device
from torch.testing import assert_close

from open_sora.modeling.dit import (
    CrossAttention,
    FastSeqParallelCrossAttention,
    SeqParallelCrossAttention,
)
from open_sora.utils.comm import gather_seq, split_seq


@parameterize("layer_cls", [SeqParallelCrossAttention, FastSeqParallelCrossAttention])
@parameterize("overlap", [True, False])
def check_sp_attn(layer_cls, overlap):
    if overlap:
        return
    model_kwargs = {}
    if layer_cls == FastSeqParallelCrossAttention:
        model_kwargs["overlap"] = overlap
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
    parallel_attn = layer_cls(
        q_dim,
        context_dim,
        num_heads,
        head_dim,
        seq_parallel_group=dist.group.WORLD,
        **model_kwargs
    ).to(get_current_device())
    parallel_attn.load_state_dict(attn.state_dict())
    hidden_states = torch.rand(bs, sq, q_dim, device=get_current_device())
    context = torch.rand(bs, skv, context_dim, device=get_current_device())
    mask = torch.zeros(bs, 1, sq, skv, device=get_current_device())
    target = attn(hidden_states, context, mask)
    hidden_states_parallel = split_seq(hidden_states, sp_size, sp_rank)
    context_parallel = split_seq(context, sp_size, sp_rank)
    output_parallel = parallel_attn(
        hidden_states_parallel,
        context_parallel,
        mask,
    )
    assert torch.equal(target.chunk(sp_size, dim=1)[sp_rank], output_parallel)
    output = gather_seq(output_parallel, sp_size, sp_rank, dist.group.WORLD)
    assert torch.equal(target, output)
    target.mean().backward()
    output.mean().backward()

    # all-reduce mean of grads
    for p in parallel_attn.parameters():
        p.grad.data.div_(sp_size)
        dist.all_reduce(p.grad.data)

    for p1, p2 in zip(attn.parameters(), parallel_attn.parameters()):
        assert_close(p1.grad, p2.grad)


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
