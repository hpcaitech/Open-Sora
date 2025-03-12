from collections import OrderedDict, defaultdict
from typing import Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler

from opensora.utils.logger import log_message
from opensora.utils.misc import format_numel_str

from .aspect import get_num_pexels_from_name
from .bucket import Bucket
from .datasets import VideoTextDataset
from .parallel import pandarallel
from .utils import sync_object_across_devices


# use pandarallel to accelerate bucket processing
# NOTE: pandarallel should only access local variables
def apply(data, method=None, seed=None, num_bucket=None, fps_max=16):
    return method(
        data["num_frames"],
        data["height"],
        data["width"],
        data["fps"],
        data["path"],
        seed + data["id"] * num_bucket,
        fps_max,
    )


class StatefulDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
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
        dataset: VideoTextDataset,
        bucket_config: dict,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        verbose: bool = False,
        num_bucket_build_workers: int = 1,
        num_groups: int = 1,
    ) -> None:
        super().__init__(
            dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last
        )
        self.dataset = dataset
        assert dataset.bucket_class == "Bucket", "Only support Bucket class for now"
        self.bucket = Bucket(bucket_config)
        self.verbose = verbose
        self.last_micro_batch_access_index = 0
        self.num_bucket_build_workers = num_bucket_build_workers
        self._cached_bucket_sample_dict = None
        self._cached_num_total_batch = None
        self.num_groups = num_groups

        if dist.get_rank() == 0:
            pandarallel.initialize(
                nb_workers=self.num_bucket_build_workers,
                progress_bar=False,
                verbose=0,
                use_memory_fs=False,
            )

    def __iter__(self) -> Iterator[list[int]]:
        bucket_sample_dict, _ = self.group_by_bucket()
        self.clear_cache()

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
        return self.get_num_batch() // self.num_groups

    def get_num_batch(self) -> int:
        _, num_total_batch = self.group_by_bucket()
        return num_total_batch

    def clear_cache(self):
        self._cached_bucket_sample_dict = None
        self._cached_num_total_batch = 0

    def group_by_bucket(self) -> dict:
        """
        Group the dataset samples into buckets.
        This method will set `self._cached_bucket_sample_dict` to the bucket sample dict.

        Returns:
            dict: a dictionary with bucket id as key and a list of sample indices as value
        """
        if self._cached_bucket_sample_dict is not None:
            return self._cached_bucket_sample_dict, self._cached_num_total_batch

        # use pandarallel to accelerate bucket processing
        log_message("Building buckets using %d workers...", self.num_bucket_build_workers)
        bucket_ids = None
        if dist.get_rank() == 0:
            data = self.dataset.data.copy(deep=True)
            data["id"] = data.index
            bucket_ids = data.parallel_apply(
                apply,
                axis=1,
                method=self.bucket.get_bucket_id,
                seed=self.seed + self.epoch,
                num_bucket=self.bucket.num_bucket,
                fps_max=self.dataset.fps_max,
            )
        dist.barrier()
        bucket_ids = sync_object_across_devices(bucket_ids)
        dist.barrier()

        # group by bucket
        # each data sample is put into a bucket with a similar image/video size
        bucket_sample_dict = defaultdict(list)
        bucket_ids_np = np.array(bucket_ids)
        valid_indices = np.where(bucket_ids_np != None)[0]
        for i in valid_indices:
            bucket_sample_dict[bucket_ids_np[i]].append(i)

        # cache the bucket sample dict
        self._cached_bucket_sample_dict = bucket_sample_dict

        # num total batch
        num_total_batch = self.print_bucket_info(bucket_sample_dict)
        self._cached_num_total_batch = num_total_batch

        return bucket_sample_dict, num_total_batch

    def print_bucket_info(self, bucket_sample_dict: dict) -> int:
        # collect statistics
        num_total_samples = num_total_batch = 0
        num_total_img_samples = num_total_vid_samples = 0
        num_total_img_batch = num_total_vid_batch = 0
        num_total_vid_batch_256 = num_total_vid_batch_768 = 0
        num_aspect_dict = defaultdict(lambda: [0, 0])
        num_hwt_dict = defaultdict(lambda: [0, 0])
        for k, v in bucket_sample_dict.items():
            size = len(v)
            num_batch = size // self.bucket.get_batch_size(k[:-1])

            num_total_samples += size
            num_total_batch += num_batch

            if k[1] == 1:
                num_total_img_samples += size
                num_total_img_batch += num_batch
            else:
                if k[0] == "256px":
                    num_total_vid_batch_256 += num_batch
                elif k[0] == "768px":
                    num_total_vid_batch_768 += num_batch
                num_total_vid_samples += size
                num_total_vid_batch += num_batch

            num_aspect_dict[k[-1]][0] += size
            num_aspect_dict[k[-1]][1] += num_batch
            num_hwt_dict[k[:-1]][0] += size
            num_hwt_dict[k[:-1]][1] += num_batch

        # sort
        num_aspect_dict = dict(sorted(num_aspect_dict.items(), key=lambda x: x[0]))
        num_hwt_dict = dict(
            sorted(num_hwt_dict.items(), key=lambda x: (get_num_pexels_from_name(x[0][0]), x[0][1]), reverse=True)
        )
        num_hwt_img_dict = {k: v for k, v in num_hwt_dict.items() if k[1] == 1}
        num_hwt_vid_dict = {k: v for k, v in num_hwt_dict.items() if k[1] > 1}

        # log
        if dist.get_rank() == 0 and self.verbose:
            log_message("Bucket Info:")
            log_message("Bucket [#sample, #batch] by aspect ratio:")
            for k, v in num_aspect_dict.items():
                log_message("(%s): #sample: %s, #batch: %s", k, format_numel_str(v[0]), format_numel_str(v[1]))
            log_message("===== Image Info =====")
            log_message("Image Bucket by HxWxT:")
            for k, v in num_hwt_img_dict.items():
                log_message("%s: #sample: %s, #batch: %s", k, format_numel_str(v[0]), format_numel_str(v[1]))
            log_message("--------------------------------")
            log_message(
                "#image sample: %s, #image batch: %s",
                format_numel_str(num_total_img_samples),
                format_numel_str(num_total_img_batch),
            )
            log_message("===== Video Info =====")
            log_message("Video Bucket by HxWxT:")
            for k, v in num_hwt_vid_dict.items():
                log_message("%s: #sample: %s, #batch: %s", k, format_numel_str(v[0]), format_numel_str(v[1]))
            log_message("--------------------------------")
            log_message(
                "#video sample: %s, #video batch: %s",
                format_numel_str(num_total_vid_samples),
                format_numel_str(num_total_vid_batch),
            )
            log_message("===== Summary =====")
            log_message("#non-empty buckets: %s", len(bucket_sample_dict))
            log_message(
                "Img/Vid sample ratio: %.2f",
                num_total_img_samples / num_total_vid_samples if num_total_vid_samples > 0 else 0,
            )
            log_message(
                "Img/Vid batch ratio: %.2f", num_total_img_batch / num_total_vid_batch if num_total_vid_batch > 0 else 0
            )
            log_message(
                "vid batch 256: %s, vid batch 768: %s", format_numel_str(num_total_vid_batch_256), format_numel_str(num_total_vid_batch_768)
            )
            log_message(
                "Vid batch ratio (256px/768px): %.2f", num_total_vid_batch_256 / num_total_vid_batch_768 if num_total_vid_batch_768 > 0 else 0
            )
            log_message(
                "#training sample: %s, #training batch: %s",
                format_numel_str(num_total_samples),
                format_numel_str(num_total_batch),
            )
        return num_total_batch

    def reset(self):
        self.last_micro_batch_access_index = 0

    def set_step(self, start_step: int):
        self.last_micro_batch_access_index = start_step * self.num_replicas

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
