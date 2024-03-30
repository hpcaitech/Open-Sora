import random
from typing import Iterator, Optional

import numpy as np
import torch
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .bucket import Bucket
from .sampler import DistributedVariableVideoSampler, VariableVideoBatchSampler


class StatefulDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index: int = 0

    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)
        indices = indices[self.start_index :]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def set_start_index(self, start_index: int) -> None:
        self.start_index = start_index


def prepare_dataloader(
    dataset,
    batch_size,
    shuffle=False,
    seed=1024,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    process_group: Optional[ProcessGroup] = None,
    **kwargs,
):
    r"""
    Prepare a dataloader for distributed training. The dataloader will be wrapped by
    `torch.utils.data.DataLoader` and `StatefulDistributedSampler`.


    Args:
        dataset (`torch.utils.data.Dataset`): The dataset to be loaded.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        seed (int, optional): Random worker seed for sampling, defaults to 1024.
        add_sampler: Whether to add ``DistributedDataParallelSampler`` to the dataset. Defaults to True.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
        pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
        num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
        kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``, more details could be found in
                `DataLoader <https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader>`_.

    Returns:
        :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
    """
    _kwargs = kwargs.copy()
    process_group = process_group or _get_default_group()
    sampler = StatefulDistributedSampler(
        dataset,
        num_replicas=process_group.size(),
        rank=process_group.rank(),
        shuffle=shuffle,
    )

    # Deterministic dataloader
    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        worker_init_fn=seed_worker,
        drop_last=drop_last,
        pin_memory=pin_memory,
        num_workers=num_workers,
        **_kwargs,
    )


class _VariableVideoBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, sampler, batch_size, drop_last, dataset, buckect_config):
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.bucket = Bucket(buckect_config)
        self.frame_interval = self.dataset.frame_interval
        self.bucket.info_bucket(self.dataset, self.frame_interval)

    def __iter__(self):
        for idx in self.sampler:
            T, H, W = self.dataset.get_data_info(idx)
            bucket_id = self.bucket.get_bucket_id(T, H, W, self.frame_interval)
            if bucket_id is None:
                continue
            rT, rH, rW = self.bucket.get_thw(bucket_id)
            self.dataset.set_data_info(idx, rT, rH, rW)
            buffer = self.bucket[bucket_id]
            buffer.append(idx)
            if len(buffer) >= self.bucket.get_batch_size(bucket_id):
                yield buffer
                self.bucket.set_empty(bucket_id)

        for k1, v1 in self.bucket.bucket.items():
            for k2, v2 in v1.items():
                for k3, buffer in v2.items():
                    if len(buffer) > 0 and not self.drop_last:
                        yield buffer
                        self.bucket.set_empty((k1, k2, k3))


def prepare_variable_dataloader(
    dataset,
    batch_size,
    bucket_config,
    shuffle=False,
    seed=1024,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    process_group=None,
    **kwargs,
):
    _kwargs = kwargs.copy()
    process_group = process_group or _get_default_group()
    sampler = DistributedVariableVideoSampler(
        dataset,
        bucket_config,
        num_replicas=process_group.size(),
        rank=process_group.rank(),
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
        verbose=True,
    )
    batch_sampler = VariableVideoBatchSampler(sampler)

    # Deterministic dataloader
    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        worker_init_fn=seed_worker,
        pin_memory=pin_memory,
        num_workers=num_workers,
        **_kwargs,
    )
