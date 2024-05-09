import os

import numpy as np
import torch
import torchvision
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

from opensora.registry import DATASETS

from .utils import VID_EXTENSIONS, get_transforms_image, get_transforms_video, read_file, temporal_random_crop

IMG_FPS = 120


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
        transform_name="center",
    ):
        self.data_path = data_path
        self.data = read_file(data_path)
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = {
            "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(transform_name, image_size),
        }

    def _print_data_number(self):
        num_videos = 0
        num_images = 0
        for path in self.data["path"]:
            if self.get_type(path) == "video":
                num_videos += 1
            else:
                num_images += 1
        print(f"Dataset contains {num_videos} videos and {num_images} images.")

    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
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
            video = temporal_random_crop(vframes, self.num_frames, self.frame_interval)

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
                path = self.data.iloc[index]["path"]
                print(f"data {path}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data)


@DATASETS.register_module()
class VariableVideoTextDataset(VideoTextDataset):
    def __init__(
        self,
        data_path,
        num_frames=None,
        frame_interval=1,
        image_size=None,
        transform_name=None,
    ):
        super().__init__(data_path, num_frames, frame_interval, image_size, transform_name=None)
        self.transform_name = transform_name
        self.data["id"] = np.arange(len(self.data))

    def get_data_info(self, index):
        T = self.data.iloc[index]["num_frames"]
        H = self.data.iloc[index]["height"]
        W = self.data.iloc[index]["width"]
        return T, H, W

    def getitem(self, index):
        # a hack to pass in the (time, height, width) info from sampler
        index, num_frames, height, width = [int(val) for val in index.split("-")]

        sample = self.data.iloc[index]
        path = sample["path"]
        text = sample["text"]
        file_type = self.get_type(path)
        ar = height / width

        video_fps = 24  # default fps
        if file_type == "video":
            # loading
            vframes, _, infos = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
            if "video_fps" in infos:
                video_fps = infos["video_fps"]

            # Sampling video frames
            video = temporal_random_crop(vframes, num_frames, self.frame_interval)

            # transform
            transform = get_transforms_video(self.transform_name, (height, width))
            video = transform(video)  # T C H W
        else:
            # loading
            image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = get_transforms_image(self.transform_name, (height, width))
            image = transform(image)

            # repeat
            video = image.unsqueeze(0)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        return {
            "video": video,
            "text": text,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
        }
