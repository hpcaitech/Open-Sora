import torch.distributed as dist


# Print debug information on selected rank
def print_rank(var_name, var_value, rank=0):
    if dist.get_rank() == rank:
        print(f"[Rank {rank}] {var_name}: {var_value}")
