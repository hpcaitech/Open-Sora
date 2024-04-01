import warnings
from collections import OrderedDict, defaultdict
from pprint import pprint
from typing import Iterator, List, Optional

import torch
from torch.utils.data import DistributedSampler

from .bucket import Bucket
from .datasets import VariableVideoTextDataset


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
    ) -> None:
        super().__init__(
            dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last
        )
        self.dataset = dataset
        self.bucket = Bucket(bucket_config)
        self.last_bucket_access_index = 0
        self.bucket_last_consumed = OrderedDict()
        self.verbose = verbose

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        bucket_sample_dict = OrderedDict()
        bucket_batch_count = OrderedDict()

        # group by bucket
        # each data sample is put into a bucket with a similar image/video size
        for i in range(len(self.dataset)):
            t, h, w = self.dataset.get_data_info(i)
            bucket_id = self.bucket.get_bucket_id(t, h, w, self.dataset.frame_interval, g)
            if bucket_id is None:
                continue
            if bucket_id not in bucket_sample_dict:
                bucket_sample_dict[bucket_id] = []
            bucket_sample_dict[bucket_id].append(i)

        # print bucket info
        if self.verbose:
            self._print_bucket_info(bucket_sample_dict)

        # process the samples
        for bucket_id, samples in bucket_sample_dict.items():
            # handle droplast
            bs_per_gpu = self.bucket.get_batch_size(bucket_id)
            global_batch_size = self.num_replicas * bs_per_gpu
            remainder = len(samples) % global_batch_size

            if remainder > 0:
                if not self.drop_last:
                    # if there is remainder, we pad to make it divisible
                    samples += samples[: global_batch_size - remainder]
                else:
                    # we just drop the remainder to make it divisible
                    samples = samples[:-remainder]
            bucket_sample_dict[bucket_id] = samples

            # handle shuffle
            if self.shuffle:
                sample_indices = torch.randperm(len(samples), generator=g).tolist()
                samples = [samples[i] for i in sample_indices]
                bucket_sample_dict[bucket_id] = samples

            # compute how many batches each bucket has
            num_batches = len(samples) // global_batch_size
            bucket_batch_count[bucket_id] = num_batches

        # compute the bucket access order
        # each bucket may have more than one batch of data
        # thus bucket_id may appear more than 1 time
        bucket_id_access_order = []
        for bucket_id, num_batch in bucket_batch_count.items():
            bucket_id_access_order.extend([bucket_id] * num_batch)

        # randomize the access order
        if self.shuffle:
            bucket_id_access_order_indices = torch.randperm(len(bucket_id_access_order), generator=g).tolist()
            bucket_id_access_order = [bucket_id_access_order[i] for i in bucket_id_access_order_indices]

        # prepare each batch from its bucket
        # according to the predefined bucket access order
        for i in range(self.last_bucket_access_index, len(bucket_id_access_order)):
            bucket_id = bucket_id_access_order[i]
            samples = bucket_sample_dict[bucket_id]

            # get the last consumed index
            last_consumed_index = self.bucket_last_consumed.get(bucket_id, 0)

            # get batch size per GPU
            bs = self.bucket.get_batch_size(bucket_id)
            total_bs = bs * self.num_replicas

            # get the current batch
            cur_batch = samples[last_consumed_index : last_consumed_index + total_bs]

            # update state dict
            if bucket_id in self.bucket_last_consumed:
                self.bucket_last_consumed[bucket_id] += total_bs
            else:
                self.bucket_last_consumed[bucket_id] = total_bs
            self.last_bucket_access_index = i

            # shard by DP
            cur_batch_on_this_rank = cur_batch[self.rank : len(cur_batch) : self.num_replicas]
            real_t, real_h, real_w = self.bucket.get_thw(bucket_id)

            # encode t, h, w into the sample index
            cur_batch_on_this_rank = [f"{idx}-{real_t}-{real_h}-{real_w}" for idx in cur_batch_on_this_rank]
            yield cur_batch_on_this_rank

        self._reset()

    def _reset(self) -> None:
        self.last_bucket_access_index = 0
        self.bucket_last_consumed = OrderedDict()

    def state_dict(self) -> dict:
        # users must ensure bucket config is the same
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "last_bucket_access_index": self.last_bucket_access_index,
            "bucket_last_consumed": self.bucket_last_consumed,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)

    def _print_bucket_info(self, bucket_sample_dict: dict) -> None:
        total_samples = 0
        num_dict = {}
        num_aspect_dict = defaultdict(int)
        num_hwt_dict = defaultdict(int)
        for k, v in bucket_sample_dict.items():
            size = len(v)
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

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)

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
            return len(self.dataset) // min_batch_size
        else:
            return (len(self.dataset) + min_batch_size - 1) // min_batch_size
