import random
from typing import Optional

import numpy as np
import torch
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils.data import DataLoader

from .datasets import VariableVideoTextDataset, VideoTextDataset
from .sampler import StatefulDistributedSampler, VariableVideoBatchSampler


# Deterministic dataloader
def get_seed_worker(seed):
    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


def prepare_dataloader(
    dataset,
    batch_size=None,
    shuffle=False,
    seed=1024,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    process_group: Optional[ProcessGroup] = None,
    bucket_config=None,
    num_bucket_build_workers=1,
    **kwargs,
):
    _kwargs = kwargs.copy()
    if isinstance(dataset, VariableVideoTextDataset):
        batch_sampler = VariableVideoBatchSampler(
            dataset,
            bucket_config,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            verbose=True,
            num_bucket_build_workers=num_bucket_build_workers,
        )
        return (
            DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                worker_init_fn=get_seed_worker(seed),
                pin_memory=pin_memory,
                num_workers=num_workers,
                **_kwargs,
            ),
            batch_sampler,
        )
    elif isinstance(dataset, VideoTextDataset):
        process_group = process_group or _get_default_group()
        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
        )
        return (
            DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                worker_init_fn=get_seed_worker(seed),
                drop_last=drop_last,
                pin_memory=pin_memory,
                num_workers=num_workers,
                **_kwargs,
            ),
            sampler,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")
