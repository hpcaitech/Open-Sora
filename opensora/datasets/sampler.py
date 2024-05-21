from collections import OrderedDict, defaultdict
from pprint import pformat
from typing import Iterator, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler

from opensora.utils.misc import format_numel_str, get_logger

from .aspect import get_num_pixels
from .bucket import Bucket
from .datasets import VariableVideoTextDataset


# use pandarallel to accelerate bucket processing
# NOTE: pandarallel should only access local variables
def apply(data, method=None, frame_interval=None, seed=None, num_bucket=None):
    return method(
        data["num_frames"],
        data["height"],
        data["width"],
        frame_interval,
        seed + data["id"] * num_bucket,
    )


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

    def reset(self) -> None:
        self.start_index = 0

    def state_dict(self, step) -> dict:
        return {"start_index": step}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)


class VariableVideoBatchSampler(DistributedSampler):
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
        num_bucket_build_workers: int = 1,
    ) -> None:
        super().__init__(
            dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last
        )
        self.dataset = dataset
        self.bucket = Bucket(bucket_config)
        self.verbose = verbose
        self.last_micro_batch_access_index = 0
        self.approximate_num_batch = None

        self._get_num_batch_cached_bucket_sample_dict = None
        self.num_bucket_build_workers = num_bucket_build_workers

    def __iter__(self) -> Iterator[List[int]]:
        if self._get_num_batch_cached_bucket_sample_dict is not None:
            bucket_sample_dict = self._get_num_batch_cached_bucket_sample_dict
            self._get_num_batch_cached_bucket_sample_dict = None
        else:
            bucket_sample_dict = self.group_by_bucket()
            if self.verbose:
                self._print_bucket_info(bucket_sample_dict)

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        bucket_micro_batch_count = OrderedDict()
        bucket_last_consumed = OrderedDict()

        # process the samples
        for bucket_id, data_list in bucket_sample_dict.items():
            # handle droplast
            bs_per_gpu = self.bucket.get_batch_size(bucket_id)
            remainder = len(data_list) % bs_per_gpu

            if remainder > 0:
                if not self.drop_last:
                    # if there is remainder, we pad to make it divisible
                    data_list += data_list[: bs_per_gpu - remainder]
                else:
                    # we just drop the remainder to make it divisible
                    data_list = data_list[:-remainder]
            bucket_sample_dict[bucket_id] = data_list

            # handle shuffle
            if self.shuffle:
                data_indices = torch.randperm(len(data_list), generator=g).tolist()
                data_list = [data_list[i] for i in data_indices]
                bucket_sample_dict[bucket_id] = data_list

            # compute how many micro-batches each bucket has
            num_micro_batches = len(data_list) // bs_per_gpu
            bucket_micro_batch_count[bucket_id] = num_micro_batches

        # compute the bucket access order
        # each bucket may have more than one batch of data
        # thus bucket_id may appear more than 1 time
        bucket_id_access_order = []
        for bucket_id, num_micro_batch in bucket_micro_batch_count.items():
            bucket_id_access_order.extend([bucket_id] * num_micro_batch)

        # randomize the access order
        if self.shuffle:
            bucket_id_access_order_indices = torch.randperm(len(bucket_id_access_order), generator=g).tolist()
            bucket_id_access_order = [bucket_id_access_order[i] for i in bucket_id_access_order_indices]

        # make the number of bucket accesses divisible by dp size
        remainder = len(bucket_id_access_order) % self.num_replicas
        if remainder > 0:
            if self.drop_last:
                bucket_id_access_order = bucket_id_access_order[: len(bucket_id_access_order) - remainder]
            else:
                bucket_id_access_order += bucket_id_access_order[: self.num_replicas - remainder]

        # prepare each batch from its bucket
        # according to the predefined bucket access order
        num_iters = len(bucket_id_access_order) // self.num_replicas
        start_iter_idx = self.last_micro_batch_access_index // self.num_replicas

        # re-compute the micro-batch consumption
        # this is useful when resuming from a state dict with a different number of GPUs
        self.last_micro_batch_access_index = start_iter_idx * self.num_replicas
        for i in range(self.last_micro_batch_access_index):
            bucket_id = bucket_id_access_order[i]
            bucket_bs = self.bucket.get_batch_size(bucket_id)
            if bucket_id in bucket_last_consumed:
                bucket_last_consumed[bucket_id] += bucket_bs
            else:
                bucket_last_consumed[bucket_id] = bucket_bs

        for i in range(start_iter_idx, num_iters):
            bucket_access_list = bucket_id_access_order[i * self.num_replicas : (i + 1) * self.num_replicas]
            self.last_micro_batch_access_index += self.num_replicas

            # compute the data samples consumed by each access
            bucket_access_boundaries = []
            for bucket_id in bucket_access_list:
                bucket_bs = self.bucket.get_batch_size(bucket_id)
                last_consumed_index = bucket_last_consumed.get(bucket_id, 0)
                bucket_access_boundaries.append([last_consumed_index, last_consumed_index + bucket_bs])

                # update consumption
                if bucket_id in bucket_last_consumed:
                    bucket_last_consumed[bucket_id] += bucket_bs
                else:
                    bucket_last_consumed[bucket_id] = bucket_bs

            # compute the range of data accessed by each GPU
            bucket_id = bucket_access_list[self.rank]
            boundary = bucket_access_boundaries[self.rank]
            cur_micro_batch = bucket_sample_dict[bucket_id][boundary[0] : boundary[1]]

            # encode t, h, w into the sample index
            real_t, real_h, real_w = self.bucket.get_thw(bucket_id)
            cur_micro_batch = [f"{idx}-{real_t}-{real_h}-{real_w}" for idx in cur_micro_batch]
            yield cur_micro_batch

        self.reset()

    def __len__(self) -> int:
        return self.get_num_batch() // dist.get_world_size()

    def group_by_bucket(self) -> dict:
        bucket_sample_dict = OrderedDict()

        from pandarallel import pandarallel

        pandarallel.initialize(nb_workers=self.num_bucket_build_workers, progress_bar=False)
        get_logger().info("Building buckets...")
        bucket_ids = self.dataset.data.parallel_apply(
            apply,
            axis=1,
            method=self.bucket.get_bucket_id,
            frame_interval=self.dataset.frame_interval,
            seed=self.seed + self.epoch,
            num_bucket=self.bucket.num_bucket,
        )

        # group by bucket
        # each data sample is put into a bucket with a similar image/video size
        for i in range(len(self.dataset)):
            bucket_id = bucket_ids[i]
            if bucket_id is None:
                continue
            if bucket_id not in bucket_sample_dict:
                bucket_sample_dict[bucket_id] = []
            bucket_sample_dict[bucket_id].append(i)
        return bucket_sample_dict

    def get_num_batch(self) -> int:
        bucket_sample_dict = self.group_by_bucket()
        self._get_num_batch_cached_bucket_sample_dict = bucket_sample_dict

        # calculate the number of batches
        if self.verbose:
            self._print_bucket_info(bucket_sample_dict)
        return self.approximate_num_batch

    def _print_bucket_info(self, bucket_sample_dict: dict) -> None:
        # collect statistics
        total_samples = 0
        total_batch = 0
        num_aspect_dict = defaultdict(lambda: [0, 0])
        num_hwt_dict = defaultdict(lambda: [0, 0])
        for k, v in bucket_sample_dict.items():
            size = len(v)
            num_batch = size // self.bucket.get_batch_size(k[:-1])

            total_samples += size
            total_batch += num_batch

            num_aspect_dict[k[-1]][0] += size
            num_aspect_dict[k[-1]][1] += num_batch
            num_hwt_dict[k[:-1]][0] += size
            num_hwt_dict[k[:-1]][1] += num_batch

        # sort
        num_aspect_dict = dict(sorted(num_aspect_dict.items(), key=lambda x: x[0]))
        num_hwt_dict = dict(
            sorted(num_hwt_dict.items(), key=lambda x: (get_num_pixels(x[0][0]), x[0][1]), reverse=True)
        )
        num_hwt_img_dict = {k: v for k, v in num_hwt_dict.items() if k[1] == 1}
        num_hwt_vid_dict = {k: v for k, v in num_hwt_dict.items() if k[1] > 1}

        # log
        if dist.get_rank() == 0 and self.verbose:
            get_logger().info("Bucket Info:")
            get_logger().info(
                "Bucket [#sample, #batch] by aspect ratio:\n%s", pformat(num_aspect_dict, sort_dicts=False)
            )
            get_logger().info(
                "Image Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_hwt_img_dict, sort_dicts=False)
            )
            get_logger().info(
                "Video Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_hwt_vid_dict, sort_dicts=False)
            )
            get_logger().info(
                "#training batch: %s, #training sample: %s, #non empty bucket: %s",
                format_numel_str(total_batch),
                format_numel_str(total_samples),
                len(bucket_sample_dict),
            )
        self.approximate_num_batch = total_batch

    def reset(self):
        self.last_micro_batch_access_index = 0

    def state_dict(self, num_steps: int) -> dict:
        # the last_micro_batch_access_index in the __iter__ is often
        # not accurate during multi-workers and data prefetching
        # thus, we need the user to pass the actual steps which have been executed
        # to calculate the correct last_micro_batch_access_index
        return {"seed": self.seed, "epoch": self.epoch, "last_micro_batch_access_index": num_steps * self.num_replicas}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)


class BatchDistributedSampler(DistributedSampler):
    """
    Used with BatchDataset;
    Suppose len_buffer == 5, num_buffers == 6, #GPUs == 3, then
           | buffer {i}          | buffer {i+1}
    ------ | ------------------- | -------------------
    rank 0 |  0,  1,  2,  3,  4, |  5,  6,  7,  8,  9
    rank 1 | 10, 11, 12, 13, 14, | 15, 16, 17, 18, 19
    rank 2 | 20, 21, 22, 23, 24, | 25, 26, 27, 28, 29
    """

    def __init__(self, dataset: Dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self.start_index = 0

    def __iter__(self):
        num_buffers = self.dataset.num_buffers
        len_buffer = self.dataset.len_buffer
        num_buffers_i = num_buffers // self.num_replicas
        num_samples_i = len_buffer * num_buffers_i

        indices_i = np.arange(self.start_index, num_samples_i) + self.rank * num_samples_i
        indices_i = indices_i.tolist()

        return iter(indices_i)

    def reset(self):
        self.start_index = 0

    def state_dict(self, step) -> dict:
        return {"start_index": step}

    def load_state_dict(self, state_dict: dict):
        self.start_index = state_dict["start_index"] + 1
