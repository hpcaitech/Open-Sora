import colossalai
import torch
import torch.distributed as dist
from colossalai.testing import spawn

from opensora.acceleration.communications import gather_forward_split_backward, split_forward_gather_backward
from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.models.layers.blocks import (
    Attention,
    MultiHeadCrossAttention,
    SeqParallelAttention,
    SeqParallelMultiHeadCrossAttention,
)


def run_attention(rank, world_size):
    # create model
    torch.manual_seed(1024)
    set_sequence_parallel_group(dist.group.WORLD)

    seq_parallel_attention = SeqParallelAttention(dim=256, num_heads=4, qkv_bias=True, enable_flash_attn=False).cuda()

    torch.manual_seed(1024)
    attention = Attention(
        dim=256,
        num_heads=4,
        qkv_bias=True,
        enable_flash_attn=False,
    ).cuda()

    # create inputs
    torch.manual_seed(1024)
    x = torch.randn(4, 64, 256).cuda()
    seq_x = x.clone().detach()

    x.requires_grad = True
    x.retain_grad()
    seq_x.requires_grad = True
    seq_x.retain_grad()

    sub_seq_x = split_forward_gather_backward(seq_x, dist.group.WORLD, dim=1, grad_scale="down")

    # run model
    out = attention(x)
    sub_seq_out = seq_parallel_attention(sub_seq_x)
    seq_out = gather_forward_split_backward(sub_seq_out, dist.group.WORLD, dim=1, grad_scale="up")

    assert torch.allclose(seq_out, out, atol=1e-7), f"{seq_out}\nvs\n{out}"

    # run backward
    seq_out.mean().backward()
    out.mean().backward()

    # all reduce gradient for sp
    for p in seq_parallel_attention.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, group=dist.group.WORLD)
            p.grad.div_(world_size)

    # check grad
    for p1, p2 in zip(seq_parallel_attention.parameters(), attention.parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-7), f"{p1.grad}\nvs\n{p2.grad}"

    # check input grad
    assert torch.allclose(x.grad, seq_x.grad, atol=1e-7), f"{x.grad}\nvs\n{seq_x.grad}"


def run_cross_attention(rank, world_size):
    # create model
    torch.manual_seed(1024)
    set_sequence_parallel_group(dist.group.WORLD)
    seq_parallel_attention = (
        SeqParallelMultiHeadCrossAttention(
            d_model=256,
            num_heads=4,
        )
        .cuda()
        .to(torch.bfloat16)
    )

    torch.manual_seed(1024)
    attention = (
        MultiHeadCrossAttention(
            d_model=256,
            num_heads=4,
        )
        .cuda()
        .to(torch.bfloat16)
    )

    # make sure the weights are the same
    for p1, p2 in zip(seq_parallel_attention.parameters(), attention.parameters()):
        p1.data.copy_(p2.data)

    # create inputs
    torch.manual_seed(1024)
    x = torch.randn(4, 64, 256).cuda().to(torch.bfloat16)
    y = torch.randn(4, 32, 256).cuda().to(torch.bfloat16)

    mask = [2, 10, 8, 16]
    mask = None
    seq_x = x.clone().detach()
    seq_y = y.clone().detach()

    # set grad
    x.requires_grad = True
    x.retain_grad()
    seq_x.requires_grad = True
    seq_x.retain_grad()
    y.requires_grad = True
    y.retain_grad()
    seq_y.requires_grad = True
    seq_y.retain_grad()

    # split by sequence
    sub_seq_x = split_forward_gather_backward(seq_x, dist.group.WORLD, dim=1, grad_scale="down")

    # run model
    out = attention(x, y, mask)
    sub_seq_out = seq_parallel_attention(sub_seq_x, seq_y, mask)
    seq_out = gather_forward_split_backward(sub_seq_out, dist.group.WORLD, dim=1, grad_scale="up")

    assert torch.allclose(seq_out, out, rtol=1e-5, atol=1e-6), f"\n{seq_out}\nvs\n{out}"

    # run backward
    seq_out.mean().backward()
    out.mean().backward()

    # all reduce gradient for sp
    for name, p in seq_parallel_attention.named_parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, group=dist.group.WORLD)
            p.grad.div_(world_size)
        else:
            print(f"grad of {name} is None")

    # # check grad
    for p1, p2 in zip(seq_parallel_attention.named_parameters(), attention.named_parameters()):
        assert torch.allclose(
            p1[1].grad, p2[1].grad, rtol=1e-3, atol=1e-4
        ), f"\n{p1[0]}\nvs\n{p2[0]}:\n{p1[1].grad}\nvs\n{p2[1].grad}"

    # # check input grad
    assert torch.allclose(x.grad, seq_x.grad, atol=1e-7), f"{x.grad}\nvs\n{seq_x.grad}"
    assert torch.allclose(y.grad, seq_y.grad, atol=1e-7), f"{y.grad}\nvs\n{seq_y.grad}"


def run_dist(rank, world_size, port):
    colossalai.launch({}, rank=rank, world_size=world_size, host="localhost", port=port)
    # run_attention(rank, world_size)
    run_cross_attention(rank, world_size)


def test_seq_parallel_attention():
    spawn(run_dist, nprocs=2)


if __name__ == "__main__":
    test_seq_parallel_attention()
