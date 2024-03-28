import random
from collections import OrderedDict
from typing import Iterator, Optional

import numpy as np
import torch
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .aspect import ASPECT_RATIOS, get_closest_ratio


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
        dataset, num_replicas=process_group.size(), rank=process_group.rank(), shuffle=shuffle
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


def find_approximate_hw(hw, hw_dict, approx=0.8):
    for k, v in hw_dict.items():
        if hw >= v * approx:
            return k
    return None


def find_closet_smaller_bucket(t, t_dict, frame_interval):
    # process image
    if t == 1:
        if 1 in t_dict:
            return 1
        else:
            return None
    # process video
    for k, v in t_dict.items():
        if t >= v * frame_interval and v != 1:
            return k
    return None


class Bucket:
    def __init__(self, bucket_config):
        for key in bucket_config:
            assert key in ASPECT_RATIOS, f"Aspect ratio {key} not found."
        # wrap config with OrderedDict
        bucket_probs = OrderedDict()
        bucket_bs = OrderedDict()
        bucket_names = sorted(bucket_config.keys(), key=lambda x: ASPECT_RATIOS[x][0], reverse=True)
        for key in bucket_names:
            bucket_time_names = sorted(bucket_config[key].keys(), key=lambda x: x, reverse=True)
            bucket_probs[key] = OrderedDict({k: bucket_config[key][k][0] for k in bucket_time_names})
            bucket_bs[key] = OrderedDict({k: bucket_config[key][k][1] for k in bucket_time_names})

        # first level: HW
        num_bucket = 0
        bucket = dict()
        hw_criteria = dict()
        t_criteria = dict()
        ar_criteria = dict()
        for k1, v1 in bucket_probs.items():
            bucket[k1] = dict()
            hw_criteria[k1] = ASPECT_RATIOS[k1][0]
            t_criteria[k1] = dict()
            ar_criteria[k1] = dict()
            for k2, _ in v1.items():
                bucket[k1][k2] = dict()
                t_criteria[k1][k2] = k2
                ar_criteria[k1][k2] = dict()
                for k3, v3 in ASPECT_RATIOS[k1][1].items():
                    bucket[k1][k2][k3] = []
                    ar_criteria[k1][k2][k3] = v3
                    num_bucket += 1

        self.bucket_probs = bucket_probs
        self.bucket_bs = bucket_bs
        self.bucket = bucket
        self.hw_criteria = hw_criteria
        self.t_criteria = t_criteria
        self.ar_criteria = ar_criteria
        self.num_bucket = num_bucket
        print(f"Number of buckets: {num_bucket}")

    def info_bucket(self, dataset, frame_interval=1):
        infos = dict()
        infos_ar = dict()
        for i in range(len(dataset)):
            T, H, W = dataset.get_data_info(i)
            bucket_id = self.get_bucket_id(T, H, W, frame_interval)
            if bucket_id is None:
                continue
            if f"{(bucket_id[0], bucket_id[1])}" not in infos:
                infos[f"{(bucket_id[0], bucket_id[1])}"] = 0
            if f"{bucket_id[2]}" not in infos_ar:
                infos_ar[f"{bucket_id[2]}"] = 0
            infos[f"{(bucket_id[0], bucket_id[1])}"] += 1
            infos_ar[f"{bucket_id[2]}"] += 1
        print(f"Dataset contains {len(dataset)} samples.")
        print("Bucket info:", infos)
        print("Aspect ratio info:", infos_ar)

    def get_bucket_id(self, T, H, W, frame_interval=1):
        # hw
        hw = H * W
        hw_id = find_approximate_hw(hw, self.hw_criteria)
        if hw_id is None:
            return None

        # hw drops by probablity
        while True:
            # T
            T_id = find_closet_smaller_bucket(T, self.t_criteria[hw_id], frame_interval)
            if T_id is not None:
                prob = self.get_prob((hw_id, T_id))
                if random.random() < prob:
                    break
            hw_id = list(self.hw_criteria.keys()).index(hw_id)
            if hw_id == len(self.hw_criteria) - 1:
                break
            hw_id = list(self.hw_criteria.keys())[hw_id + 1]
        if T_id is None:
            return None

        # ar
        ar_criteria = self.ar_criteria[hw_id][T_id]
        ar_id = get_closest_ratio(H, W, ar_criteria)
        return hw_id, T_id, ar_id

    def get_thw(self, bucket_id):
        assert len(bucket_id) == 3
        T = self.t_criteria[bucket_id[0]][bucket_id[1]]
        H, W = self.ar_criteria[bucket_id[0]][bucket_id[1]][bucket_id[2]]
        return T, H, W

    def get_prob(self, bucket_id):
        return self.bucket_probs[bucket_id[0]][bucket_id[1]]

    def get_batch_size(self, bucket_id):
        return self.bucket_bs[bucket_id[0]][bucket_id[1]]

    def __getitem__(self, index):
        assert len(index) == 3
        return self.bucket[index[0]][index[1]][index[2]]

    def set_empty(self, index):
        assert len(index) == 3
        self.bucket[index[0]][index[1]][index[2]] = []

    def __len__(self):
        return self.num_bucket


def closet_smaller_bucket(value, bucket):
    for i in range(1, len(bucket)):
        if value < bucket[i]:
            return bucket[i - 1]
    return bucket[-1]


class VariableVideoBatchSampler(torch.utils.data.BatchSampler):
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
    sampler = StatefulDistributedSampler(
        dataset, num_replicas=process_group.size(), rank=process_group.rank(), shuffle=shuffle
    )
    batch_sampler = VariableVideoBatchSampler(
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        dataset=dataset,
        buckect_config=bucket_config,
    )

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
