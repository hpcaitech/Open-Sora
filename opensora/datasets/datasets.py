import os

import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

from opensora.registry import DATASETS

from . import video_transforms
from .utils import VID_EXTENSIONS, get_transforms_image, get_transforms_video

@DATASETS.register_module()
class VideoTextDataset(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        data_path,
        num_frames=16,
        frame_interval=1,
        image_size=(256, 256),
    ):
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
        self.transforms = {
            "image": get_transforms_image(image_size[0]),
            "video": get_transforms_video(image_size[0]),
        }

    def get_type(self, path):
        ext = path.split(".")[-1]
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert f".{ext.lower()}" in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"

    def getitem(self, index):
        sample = self.data.iloc[index]
        path = sample["path"]
        text = sample["text"]
        file_type = self.get_type(path)

        if file_type == "video":
            # loading
            vframes, _, _ = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")

            # Sampling video frames
            total_frames = len(vframes)
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert (
                end_frame_ind - start_frame_ind >= self.num_frames
            ), f"{path} with index {index} has not enough frames."
            frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)
            video = vframes[frame_indice]

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
        return {"video": video, "text": text}

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
