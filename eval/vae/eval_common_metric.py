"""Calculates the CLIP Scores

The CLIP model is a contrasitively learned language-image model. There is
an image encoder and a text encoder. It is believed that the CLIP model could
measure the similarity of cross modalities. Please find more information from
https://github.com/openai/CLIP.

The CLIP Score measures the Cosine Similarity between two embedded features.
This repository utilizes the pretrained CLIP Model to calculate
the mean average of cosine similarities.

See --help to see further details.

Code apapted from https://github.com/mseitzer/pytorch-fid and https://github.com/openai/CLIP.

Copyright 2023 The Hong Kong Polytechnic University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import os.path as osp
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
from decord import VideoReader, cpu
from pytorchvideo.transforms import ShortSideScale
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo

sys.path.append(".")
from cal_flolpips import calculate_flolpips
from cal_lpips import calculate_lpips
from cal_psnr import calculate_psnr
from cal_ssim import calculate_ssim

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


class VideoDataset(Dataset):
    def __init__(
        self,
        real_video_dir,
        generated_video_dir,
        num_frames,
        sample_rate=1,
        crop_size=None,
        resolution=128,
    ) -> None:
        super().__init__()
        self.real_video_files = self._combine_without_prefix(real_video_dir)
        self.generated_video_files = self._combine_without_prefix(generated_video_dir)
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.crop_size = crop_size
        self.short_size = resolution

    def __len__(self):
        return len(self.real_video_files)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        real_video_file = self.real_video_files[index]
        generated_video_file = self.generated_video_files[index]
        print(real_video_file, generated_video_file)
        real_video_tensor = self._load_video(real_video_file)
        generated_video_tensor = self._load_video(generated_video_file)
        return {"real": real_video_tensor, "generated": generated_video_tensor}

    def _load_video(self, video_path):
        num_frames = self.num_frames
        sample_rate = self.sample_rate
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(decord_vr)
        sample_frames_len = sample_rate * num_frames

        if total_frames >= sample_frames_len:
            s = 0
            e = s + sample_frames_len
            num_frames = num_frames
        else:
            s = 0
            e = total_frames
            num_frames = int(total_frames / sample_frames_len * num_frames)
            print(
                f"sample_frames_len {sample_frames_len}, only can sample {num_frames * sample_rate}",
                video_path,
                total_frames,
            )

        frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        return _preprocess(video_data, short_size=self.short_size, crop_size=self.crop_size)

    def _combine_without_prefix(self, folder_path, prefix="."):
        folder = []
        os.makedirs(folder_path, exist_ok=True)
        for name in os.listdir(folder_path):
            if name[0] == prefix:
                continue
            if osp.isfile(osp.join(folder_path, name)):
                folder.append(osp.join(folder_path, name))
        folder.sort()
        return folder


def _preprocess(video_data, short_size=128, crop_size=None):
    transform = Compose(
        [
            Lambda(lambda x: x / 255.0),
            ShortSideScale(size=short_size),
            CenterCropVideo(crop_size=crop_size),
        ]
    )
    video_outputs = transform(video_data)
    # video_outputs = torch.unsqueeze(video_outputs, 0) # (bz,c,t,h,w)
    return video_outputs


def calculate_common_metric(args, dataloader, device):
    metric_dict = {}
    if type(args.metric) is str:
        args.metric = [m.strip() for m in args.metric.split(",")]
    print(args.metric)
    for metric in args.metric:
        score_list = []
        for batch_data in tqdm(dataloader):  # {'real': real_video_tensor, 'generated':generated_video_tensor }
            real_videos = batch_data["real"]
            generated_videos = batch_data["generated"]
            assert real_videos.shape[2] == generated_videos.shape[2]
            if metric == "ssim":
                tmp_list = list(calculate_ssim(real_videos, generated_videos)["value"].values())
            elif metric == "psnr":
                tmp_list = list(calculate_psnr(real_videos, generated_videos)["value"].values())
            elif metric == "flolpips":
                result = calculate_flolpips(real_videos, generated_videos, args.device)
                tmp_list = list(result["value"].values())
            elif metric == "lpips":
                tmp_list = list(calculate_lpips(real_videos, generated_videos, args.device)["value"].values())
            else:
                print(f"metric {metric} is not in acceped list, not calculated")
                continue
            score_list += tmp_list
        metric_dict[metric] = np.mean(score_list)

    return metric_dict


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use")
    parser.add_argument("--real_video_dir", type=str, help=("the path of real videos`"))
    parser.add_argument("--generated_video_dir", type=str, help=("the path of generated videos`"))
    parser.add_argument("--device", type=str, default=None, help="Device to use. Like cuda, cuda:0 or cpu")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=("Number of processes to use for data loading. " "Defaults to `min(8, num_cpus)`"),
    )
    parser.add_argument("--sample_fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=336)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--subset_size", type=int, default=None)
    # parser.add_argument("--metric", type=str, default="fvd",choices=['fvd','psnr','ssim','lpips', 'flolpips'])
    parser.add_argument("--metric", nargs="+", default=[])
    parser.add_argument("--fvd_method", type=str, default="styleganv", choices=["styleganv", "videogpt"])

    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    dataset = VideoDataset(
        args.real_video_dir,
        args.generated_video_dir,
        num_frames=args.num_frames,
        sample_rate=args.sample_rate,
        crop_size=args.crop_size,
        resolution=args.resolution,
    )

    if args.subset_size:
        indices = range(args.subset_size)
        dataset = Subset(dataset, indices=indices)

    dataloader = DataLoader(dataset, args.batch_size, num_workers=num_workers, pin_memory=True)

    metric_score = calculate_common_metric(args, dataloader, device)
    print("metric: ", args.metric, " ", metric_score)


if __name__ == "__main__":
    main()
