import torch.distributed as dist
import socket

_GLOBAL_PARALLEL_GROUPS = dict()


def set_data_parallel_group(group: dist.ProcessGroup):
    _GLOBAL_PARALLEL_GROUPS["data"] = group


def get_data_parallel_group():
    return _GLOBAL_PARALLEL_GROUPS.get("data", dist.group.WORLD)

def set_cond_groups():
    world_size = dist.get_world_size()
    hostname = socket.gethostname()
    hostnames = [None] * world_size
    dist.all_gather_object(hostnames, hostname)
    hosts = set(hostnames)
    master = hostnames[0]
    if len(hosts) > 2:
        raise NotImplementedError("More than 2 hosts are not supported yet")
    elif len(hosts) == 2:
        _GLOBAL_PARALLEL_GROUPS["cond"] = dist.new_group(ranks=[i for i in range(world_size) if hostnames[i] == master])
        _GLOBAL_PARALLEL_GROUPS["uncond"] = dist.new_group(ranks=[i for i in range(world_size) if hostnames[i] != master])
    else:
        _GLOBAL_PARALLEL_GROUPS["cond"] = dist.new_group(ranks=[i for i in range(world_size // 2)])
        _GLOBAL_PARALLEL_GROUPS["uncond"] = dist.new_group(ranks=[i for i in range(world_size // 2, world_size)])
    _GLOBAL_PARALLEL_GROUPS["sequence"] = get_cond_parallel_group() if check_cond() else get_uncond_parallel_group()

def set_sequence_parallel_group(group: dist.ProcessGroup):
    _GLOBAL_PARALLEL_GROUPS["sequence"] = group


def get_sequence_parallel_group():
    return _GLOBAL_PARALLEL_GROUPS.get("sequence", None)

def get_cond_parallel_group():
    return _GLOBAL_PARALLEL_GROUPS.get("cond", dist.group.WORLD)

def get_uncond_parallel_group():
    return _GLOBAL_PARALLEL_GROUPS.get("uncond", dist.group.WORLD)

def get_cond_master():
    return dist.get_process_group_ranks(get_cond_parallel_group())[0]

def get_uncond_master():
    return dist.get_process_group_ranks(get_uncond_parallel_group())[0]

def check_cond():
    cond_ranks = dist.get_process_group_ranks(get_cond_parallel_group())
    return dist.get_rank() in cond_ranks