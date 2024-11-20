# test_dist.py
import argparse
import os
import time

import torch
import torch_musa
import torch.distributed as dist
from prettytable import PrettyTable


def init_dist():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    dist.init_process_group(world_size=world_size, rank=rank,
                            init_method="env://", backend="mccl")
    torch.musa.set_device(local_rank)


def get_time():
    torch.musa.synchronize()
    return time.time()


def log(*args):
    if dist.get_rank() == 0:
        print(*args)

### Communicatoin ops ###


class CommOp:
    def __init__(self, world_size: int):
        self.world_size = world_size

    def __call__(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError

    def bw_factor(self):
        raise NotImplementedError


class AllReduce(CommOp):
    def __call__(self, tensor: torch.Tensor):
        tensor_out = tensor.contiguous()
        dist.all_reduce(tensor_out)

    def bw_factor(self):
        return 2 * (self.world_size - 1) / self.world_size


class AllGather(CommOp):
    def __call__(self, tensor: torch.Tensor) -> None:
        # input_chunks = list(torch.chunk(tensor, self.world_size, dim=0))
        # input_chunks = torch.zeros_like(tensor)
        # dist.all_gather(input_chunks, tensor)
        
        # all tensor
        # tensor = tensor.contiguous()
        tensor_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)

        # concat
        # output = torch.cat(tensor_list, dim=0).contiguous()

    def bw_factor(self):
        return (self.world_size - 1) / self.world_size


class ReduceScatter(CommOp):
    def __call__(self, tensor: torch.Tensor) -> None:
        input_chunks = list(torch.chunk(tensor, self.world_size, dim=0))
        out = torch.empty_like(input_chunks[0])
        dist.reduce_scatter(out, input_chunks)

    def bw_factor(self):
        return (self.world_size - 1) / self.world_size


OPS = {
    'allreduce': AllReduce,
    'allgather': AllGather,
    'reducescatter': ReduceScatter,
}


def collect_time(tensor: torch.Tensor, op: CommOp, n_iters: int) -> float:
    start = get_time()
    for _ in range(n_iters):
        op(tensor)
    end = get_time()
    return (end - start) / n_iters


def benchmark(op: CommOp, sizes: list, n_iters: int, n_warmup: int = 5, dtype=torch.float) -> None:
    element_size = torch.finfo(dtype).bits // 8
    sizes = sorted(sizes)
    counts = [size // element_size for size in sizes]
    # warmup for min
    tensor = torch.rand(counts[0], dtype=dtype, device='musa')
    collect_time(tensor, op, n_warmup)
    # warmup for max
    tensor = torch.rand(counts[-1], dtype=dtype, device='musa')
    collect_time(tensor, op, n_warmup)
    # benchmark
    busbw_sum = 0
    table = PrettyTable(['size(B)', 'count(elements)', 'type', 'time(ms)',
                        'algbw(GB/s)', 'busbw(GB/s)'], float_format='.2')
    for size, count in zip(sizes, counts):
        assert size % element_size == 0, "size must be divisible by element_size"
        tensor = torch.rand(count, dtype=dtype, device='musa')
        duration = collect_time(tensor, op, n_iters)
        duration = torch.tensor([duration], device='musa')
        dist.all_reduce(duration)
        duration.div_(dist.get_world_size())
        algbw = size / duration.item()
        busbw = algbw * op.bw_factor()
        busbw_sum += busbw
        table.add_row([size, count, dtype, duration.item() *
                      1000, algbw / 1024**3, busbw / 1024**3])
    avg_busbw = busbw_sum / len(sizes)
    if dist.get_rank() == 0:
        print(table)
        print(f'Average busbw: {avg_busbw/1024**3:.3f} GB/s')


def parse_size(s: str) -> int:
    s = s.upper()
    if s[-1] == 'B':
        s = s[:-1]
    if s[-1] == 'K':
        return int(s[:-1]) * 1024
    if s[-1] == 'M':
        return int(s[:-1]) * 1024**2
    if s[-1] == 'G':
        return int(s[:-1]) * 1024**3
    return int(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', type=str, default='allreduce')
    parser.add_argument('-b', '--begin', type=str, default='32M')
    parser.add_argument('-e', '--end', type=str, default='32M')
    parser.add_argument('-s', '--step', type=str, default='2M')
    parser.add_argument('-f', '--factor', type=int, default=1)
    parser.add_argument('-i', '--iters', type=int, default=20)
    parser.add_argument('-w', '--warmup', type=int, default=5)
    parser.add_argument('-d', '--dtype', type=str,
                        default='float', choices=['float', 'fp16', 'bf16'])
    args = parser.parse_args()
    init_dist()
    comm_op = OPS[args.algorithm.lower()](dist.get_world_size())
    if args.factor > 1:
        sizes = []
        start = parse_size(args.begin)
        end = parse_size(args.end)
        while start <= end:
            sizes.append(start)
            start *= args.factor
    else:
        sizes = list(range(parse_size(args.begin), parse_size(
            args.end) + 1, parse_size(args.step)))
    if args.dtype == 'float':
        dtype = torch.float
    elif args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    benchmark(comm_op, sizes, args.iters, args.warmup, dtype=dtype)