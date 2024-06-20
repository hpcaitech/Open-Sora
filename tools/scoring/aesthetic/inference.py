# adapted from https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/simple_inference.py
import cv2  # isort:skip

import argparse
import gc
import os
from datetime import timedelta

import clip
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets.folder import pil_loader
from tqdm import tqdm

from tools.datasets.utils import extract_frames, is_video

NUM_FRAMES_POINTS = {
    1: (0.5,),
    2: (0.25, 0.5),
    3: (0.1, 0.5, 0.9),
}


def merge_scores(gathered_list: list, meta: pd.DataFrame, column):
    # reorder
    indices_list = list(map(lambda x: x[0], gathered_list))
    scores_list = list(map(lambda x: x[1], gathered_list))

    flat_indices = []
    for x in zip(*indices_list):
        flat_indices.extend(x)
    flat_scores = []
    for x in zip(*scores_list):
        flat_scores.extend(x)
    flat_indices = np.array(flat_indices)
    flat_scores = np.array(flat_scores)

    # filter duplicates
    unique_indices, unique_indices_idx = np.unique(flat_indices, return_index=True)
    meta.loc[unique_indices, column] = flat_scores[unique_indices_idx]

    # drop indices in meta not in unique_indices
    meta = meta.loc[unique_indices]
    return meta


class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(self, meta_path, transform=None, num_frames=3):
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)
        self.transform = transform
        self.points = NUM_FRAMES_POINTS[num_frames]

    def __getitem__(self, index):
        sample = self.meta.iloc[index]
        path = sample["path"]

        # extract frames
        if not is_video(path):
            images = [pil_loader(path)]
        else:
            num_frames = sample["num_frames"] if "num_frames" in sample else None
            images = extract_frames(sample["path"], points=self.points, backend="opencv", num_frames=num_frames)

        # transform
        images = [self.transform(img) for img in images]

        # stack
        images = torch.stack(images)

        ret = dict(index=index, images=images)
        return ret

    def __len__(self):
        return len(self.meta)


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
        self.clip, self.preprocess = clip.load("ViT-L/14", device=device)

        self.eval()
        self.to(device)

    def forward(self, x):
        image_features = self.clip.encode_image(x)
        image_features = F.normalize(image_features, p=2, dim=-1).float()
        return self.mlp(image_features)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--bs", type=int, default=1024, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
    parser.add_argument("--prefetch_factor", type=int, default=3, help="Prefetch factor")
    parser.add_argument("--num_frames", type=int, default=3, help="Number of frames to extract")
    parser.add_argument("--skip_if_existing", action="store_true")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        exit()

    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_aes{ext}"
    if args.skip_if_existing and os.path.exists(out_path):
        print(f"Output meta file '{out_path}' already exists. Exit.")
        exit()

    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    # build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AestheticScorer(768, device)
    model.mlp.load_state_dict(torch.load("pretrained_models/aesthetic.pth", map_location=device))
    preprocess = model.preprocess

    # build dataset
    dataset = VideoTextDataset(args.meta_path, transform=preprocess, num_frames=args.num_frames)
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        sampler=DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False,
        ),
    )

    # compute aesthetic scores
    indices_list = []
    scores_list = []
    model.eval()
    for batch in tqdm(dataloader, disable=dist.get_rank() != 0):
        indices = batch["index"]
        images = batch["images"].to(device, non_blocking=True)

        B = images.shape[0]
        images = rearrange(images, "B N C H W -> (B N) C H W")

        # compute score
        with torch.no_grad():
            scores = model(images)

        scores = rearrange(scores, "(B N) 1 -> B N", B=B)
        scores = scores.mean(dim=1)
        scores_np = scores.to(torch.float32).cpu().numpy()

        indices_list.extend(indices.tolist())
        scores_list.extend(scores_np.tolist())

    # save local results
    meta_local = merge_scores([(indices_list, scores_list)], dataset.meta, column="aes")
    save_dir_local = os.path.join(os.path.dirname(out_path), "parts")
    os.makedirs(save_dir_local, exist_ok=True)
    out_path_local = os.path.join(
        save_dir_local, os.path.basename(out_path).replace(".csv", f"_part_{dist.get_rank()}.csv")
    )
    meta_local.to_csv(out_path_local, index=False)

    # wait for all ranks to finish data processing
    dist.barrier()

    torch.cuda.empty_cache()
    gc.collect()
    gathered_list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_list, (indices_list, scores_list))
    if dist.get_rank() == 0:
        meta_new = merge_scores(gathered_list, dataset.meta, column="aes")
        meta_new.to_csv(out_path, index=False)
        print(f"New meta with aesthetic scores saved to '{out_path}'.")


if __name__ == "__main__":
    main()
