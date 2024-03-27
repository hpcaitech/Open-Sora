import random

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.distributed.distributed_c10d import _get_default_group
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

from opensora.registry import DATASETS

from .utils import (
    VID_EXTENSIONS,
    StatefulDistributedSampler,
    get_transforms_image,
    get_transforms_video,
    temporal_random_crop,
)


def closet_smaller_bucket(value, bucket):
    for i in range(1, len(bucket)):
        if value < bucket[i]:
            return bucket[i - 1]
    return bucket[-1]


@DATASETS.register_module()
class VariableVideoTextDataset(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        data_path,
        num_frames=None,
        frame_interval=1,
        image_size=(256, 256),
        bucket=None,
        batch_size_bucket=None,
        transform_name="center",
    ):
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        self.batch_size_bucket = batch_size_bucket

        # build bucket
        self.bucket = bucket
        num_effect_frames = self.data["num_frames"] // frame_interval
        self.data = self.data[num_effect_frames >= self.bucket[0]]
        self.data["bucket"] = num_effect_frames.apply(lambda x: closet_smaller_bucket(x, bucket))
        gb = self.data.groupby("bucket")
        self.data_bucket = {x: gb.get_group(x) for x in bucket}
        self.data_bucket_len = {x: len(self.data_bucket[x]) for x in bucket}
        print(self.data_bucket_len)

        self.num_frames = num_frames
        assert self.num_frames is None, "num_frames must be None"
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = {
            "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(transform_name, image_size),
        }

    def get_type(self, path):
        ext = path.split(".")[-1]
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert f".{ext.lower()}" in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"

    def get_bucket(self, index):
        return self.data.iloc[index]["bucket"]

    def getitem(self, index):
        sample = self.data.iloc[index]
        path = sample["path"]
        text = sample["text"]
        file_type = self.get_type(path)
        num_frames = self.get_bucket(index)

        if file_type == "video":
            # loading
            vframes, _, _ = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")

            # Sampling video frames
            video = temporal_random_crop(vframes, num_frames, self.frame_interval)

            # transform
            transform = self.transforms["video"]
            video = transform(video)  # T C H W
        else:
            # loading
            image = pil_loader(path)

            # transform
            transform = self.transforms["image"]
            image = transform(image)

            # repeat
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        return {"video": video, "text": text, "num_frames": num_frames}

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data)


class VariableVideoBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, sampler, dataset, batch_size, bucket, batch_size_bucket, drop_last=False):
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket = bucket
        self.batch_size_bucket = batch_size_bucket
        self._buckets = {x: [] for x in bucket}
        self.drop_last = drop_last

    def __iter__(self):
        for idx in self.sampler:
            bucket_id = self.dataset.get_bucket(idx)
            bucket = self._buckets[bucket_id]
            bucket.append(idx)
            if len(bucket) >= self.batch_size_bucket[bucket_id]:
                yield bucket
                self._buckets[bucket_id] = []

        for bucket in self._buckets.values():
            if len(bucket) > 0 and not self.drop_last:
                yield bucket


def prepare_dataloader_with_batchsampler(
    dataset,
    batch_size,
    shuffle=False,
    seed=1024,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    process_group=None,
    bucket=None,
    batch_size_bucket=None,
    **kwargs,
):
    _kwargs = kwargs.copy()
    process_group = process_group or _get_default_group()
    sampler = StatefulDistributedSampler(
        dataset, num_replicas=process_group.size(), rank=process_group.rank(), shuffle=shuffle
    )
    batch_sampler = VariableVideoBatchSampler(
        sampler,
        dataset,
        batch_size=batch_size,
        bucket=bucket,
        batch_size_bucket=batch_size_bucket,
        drop_last=drop_last,
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
