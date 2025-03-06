import torch.distributed as dist

_GLOBAL_PARALLEL_GROUPS = dict()


def set_data_parallel_group(group: dist.ProcessGroup):
    _GLOBAL_PARALLEL_GROUPS["data"] = group


def get_data_parallel_group(get_mixed_dp_pg : bool = False):
    if get_mixed_dp_pg and "mixed_dp_group" in _GLOBAL_PARALLEL_GROUPS:
        return _GLOBAL_PARALLEL_GROUPS["mixed_dp_group"]
    return _GLOBAL_PARALLEL_GROUPS.get("data", dist.group.WORLD)


def set_sequence_parallel_group(group: dist.ProcessGroup):
    _GLOBAL_PARALLEL_GROUPS["sequence"] = group


def get_sequence_parallel_group():
    return _GLOBAL_PARALLEL_GROUPS.get("sequence", None)


def set_tensor_parallel_group(group: dist.ProcessGroup):
    _GLOBAL_PARALLEL_GROUPS["tensor"] = group


def get_tensor_parallel_group():
    return _GLOBAL_PARALLEL_GROUPS.get("tensor", None)
