import math
import warnings
from collections import OrderedDict, defaultdict
from pprint import pprint
from typing import Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DistributedSampler, Sampler

from .bucket import Bucket
from .datasets import VariableVideoTextDataset


class DistributedVariableVideoSampler(DistributedSampler):
    def __init__(
        self,
        dataset: VariableVideoTextDataset,
        bucket_config: dict,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.bucket = Bucket(bucket_config)
        self.last_bucket_id = None
        self.last_bucket_comsumed_samples = 0
        self.verbose = verbose

    def _reset(self) -> None:
        self.last_bucket_id = None
        self.last_bucket_comsumed_samples = 0

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self._reset()

    def __iter__(self) -> Iterator[Tuple[tuple, int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        bucket_sample_dict = OrderedDict()
        # group by bucket
        for i in range(len(self.dataset)):
            t, h, w = self.dataset.get_data_info(i)
            bucket_id = self.bucket.get_bucket_id(t, h, w, self.dataset.frame_interval, g)
            if bucket_id is None:
                continue
            real_t, real_h, real_w = self.bucket.get_thw(bucket_id)
            self.dataset.set_data_info(i, real_t, real_h, real_w)
            if bucket_id not in bucket_sample_dict:
                bucket_sample_dict[bucket_id] = []
            bucket_sample_dict[bucket_id].append(i)
        # shuffle
        if self.shuffle:
            # sort buckets
            bucket_indices = torch.randperm(len(bucket_sample_dict), generator=g).tolist()
            bucket_order = {k: bucket_indices[i] for i, k in enumerate(bucket_sample_dict)}
            # sort samples in each bucket
            for k, v in bucket_sample_dict.items():
                sample_indices = torch.randperm(len(v), generator=g).tolist()
                samples_reordered = [v[i] for i in sample_indices]
                bucket_sample_dict[k] = samples_reordered
        # all random numbers should be generated before this line
        # pad / slice each bucket
        for k, v in bucket_sample_dict.items():
            # skip last comsumed samples
            if k == self.last_bucket_id:
                v = v[self.last_bucket_comsumed_samples :]
            total_size = self._get_real_total_size(len(v))
            if not self.drop_last:
                padding_size = total_size - len(v)
                if padding_size <= len(v):
                    v += v[:padding_size]
                else:
                    v += (v * math.ceil(padding_size / len(v)))[:padding_size]
            else:
                v = v[:total_size]
            assert len(v) == total_size
            # subsample
            v = v[self.rank : total_size : self.num_replicas]
            bucket_sample_dict[k] = v
        # shuffle buckets after printing to keep the original order
        if self.verbose:
            self._print_bucket_info(bucket_sample_dict)
        if self.shuffle:
            bucket_sample_dict = OrderedDict(sorted(bucket_sample_dict.items(), key=lambda item: bucket_order[item[0]]))
        # iterate
        found_last_bucket = self.last_bucket_id is None
        for k, v in bucket_sample_dict.items():
            if k == self.last_bucket_id:
                found_last_bucket = True
            if not found_last_bucket:
                continue
            self.last_bucket_id = k
            for sample_idx in v:
                self.last_bucket_comsumed_samples += self.num_replicas
                yield k, sample_idx
            self.last_bucket_comsumed_samples = 0
        self._reset()

    def _get_real_total_size(self, size: int) -> int:
        if self.drop_last and size % self.num_replicas != 0:
            num_samples = math.ceil((size - self.num_replicas) / self.num_replicas)
        else:
            num_samples = math.ceil(size / self.num_replicas)
        total_size = num_samples * self.num_replicas
        return total_size

    def __len__(self) -> int:
        warnings.warn(
            "The length of DistributedVariableVideoSampler is dynamic and may not be accurate. Return the max value."
        )
        return len(self.dataset)

    def _print_bucket_info(self, bucket_sample_dict: dict) -> None:
        total_samples = 0
        num_dict = {}
        num_aspect_dict = defaultdict(int)
        num_hwt_dict = defaultdict(int)
        for k, v in bucket_sample_dict.items():
            size = len(v) * self.num_replicas
            total_samples += size
            num_dict[k] = size
            num_aspect_dict[k[-1]] += size
            num_hwt_dict[k[:-1]] += size
        print(f"Total training samples: {total_samples}, num buckets: {len(num_dict)}")
        print("Bucket samples:")
        pprint(num_dict)
        print("Bucket samples by HxWxT:")
        pprint(num_hwt_dict)
        print("Bucket samples by aspect ratio:")
        pprint(num_aspect_dict)

    def state_dict(self) -> dict:
        # users must ensure bucket config is the same
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "last_bucket_id": self.last_bucket_id,
            "last_bucket_comsumed_samples": self.last_bucket_comsumed_samples,
        }

    def load_state_dict(self, state_dict) -> None:
        self.__dict__.update(state_dict)


class VariableVideoBatchSampler(Sampler[List[int]]):
    def __init__(self, sampler: DistributedVariableVideoSampler) -> None:
        self.sampler = sampler
        self.drop_last = sampler.drop_last
        self.bucket = sampler.bucket

    def __iter__(self) -> Iterator[List[int]]:
        sampler_iter = iter(self.sampler)
        # init cur bucket
        try:
            cur_bucket_id, sample_idx = next(sampler_iter)
        except StopIteration:
            return
        cur_batch_size = self.bucket.get_batch_size(cur_bucket_id)
        cur_sample_indices = [sample_idx]
        # iterate the rest
        for bucket_id, sample_idx in sampler_iter:
            if len(cur_sample_indices) == cur_batch_size:
                yield cur_sample_indices
                cur_sample_indices = []
            if bucket_id != cur_bucket_id:
                if len(cur_sample_indices) > 0 and not self.drop_last:
                    yield cur_sample_indices
                cur_bucket_id = bucket_id
                cur_batch_size = self.bucket.get_batch_size(cur_bucket_id)
                cur_sample_indices = [sample_idx]
            else:
                cur_sample_indices.append(sample_idx)
        if len(cur_sample_indices) > 0 and (not self.drop_last or len(cur_sample_indices) == cur_batch_size):
            yield cur_sample_indices

    def state_dict(self) -> dict:
        return self.sampler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.sampler.load_state_dict(state_dict)

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)

    def __len__(self) -> int:
        warnings.warn(
            "The length of VariableVideoBatchSampler is dynamic and may not be accurate. Return the max value."
        )
        min_batch_size = None
        for v in self.bucket.bucket_bs.values():
            for bs in v.values():
                if bs is not None and (min_batch_size is None or bs < min_batch_size):
                    min_batch_size = bs
        if self.drop_last:
            return len(self.sampler) // min_batch_size
        else:
            return (len(self.sampler) + min_batch_size - 1) // min_batch_size
