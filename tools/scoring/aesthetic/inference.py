# adapted from https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/simple_inference.py
import argparse
from datetime import timedelta

import clip
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from colossalai.utils import set_seed
from einops import rearrange
from PIL import Image
from torchvision.datasets.folder import pil_loader
from tqdm import tqdm

from tools.datasets.utils import extract_frames, is_video

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


NUM_FRAMES_POINTS = {
    1: (0.5,),
    2: (0.25, 0.5),
    3: (0.1, 0.5, 0.9),
}


class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None, num_frames=3):
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.points = NUM_FRAMES_POINTS[num_frames]

    def getitem(self, index):
        sample = self.data.iloc[index]
        path = sample["path"]
        if not is_video(path):
            images = [pil_loader(path)]
        else:
            num_frames = None
            if "num_frames" in sample:
                num_frames = sample["num_frames"]
            images = extract_frames(sample["path"], points=self.points, backend="opencv", num_frames=num_frames)
        images = [self.transform(img) for img in images]
        images = torch.stack(images)
        ret = dict(index=index, images=images)
        return ret

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.getitem(index)


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticScorer(nn.Module):
    def __init__(self, input_size, device):
        super().__init__()
        self.mlp = MLP(input_size)
        self.mlp.load_state_dict(torch.load("pretrained_models/aesthetic.pth"))
        self.clip, self.preprocess = clip.load("ViT-L/14", device=device)

        self.eval()
        self.to(device)

    def forward(self, x):
        image_features = self.clip.encode_image(x)
        image_features = F.normalize(image_features, p=2, dim=-1).float()
        return self.mlp(image_features)


@torch.inference_mode()
def main(args):
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(1024)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    output_file = args.input.replace(".csv", f"_aes_part{rank}.csv")

    # build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AestheticScorer(768, device)
    preprocess = model.preprocess

    # build dataset
    dataset = VideoTextDataset(args.input, transform=preprocess, num_frames=args.num_frames)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    # compute aesthetic scores
    dataset.data["aes"] = np.nan

    with tqdm(dataloader, position=rank, desc=f"Data Parallel Rank {rank}") as t:
        for idx, batch in enumerate(t):
            image_indices = batch["index"]
            images = batch["images"].to(device, non_blocking=True)
            B = images.shape[0]
            images = rearrange(images, "b p c h w -> (b p) c h w")

            # compute score
            scores = model(images)
            scores = rearrange(scores, "(b p) 1 -> b p", b=B)
            scores = scores.mean(dim=1)
            scores_np = scores.to(torch.float32).cpu().numpy()

            # assign the score
            dataset.data.loc[image_indices, "aes"] = scores_np

    # wait for all ranks to finish data processing
    dist.barrier()

    # exclude rows whose aes is nan and save file
    dataset.data = dataset.data[dataset.data["aes"] > 0]
    dataset.data.to_csv(output_file, index=False)
    print(f"New meta with aesthetic scores saved to '{output_file}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to the input CSV file")
    parser.add_argument("--bs", type=int, default=1024, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
    parser.add_argument("--accumulate", type=int, default=1, help="batch to accumulate")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Prefetch factor")
    parser.add_argument("--num_frames", type=int, default=3, help="Number of frames to extract")
    args = parser.parse_args()

    main(args)
