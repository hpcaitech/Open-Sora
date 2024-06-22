import argparse
import os

import clip
import colossalai
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets.folder import pil_loader
from tqdm import tqdm

from tools.datasets.utils import extract_frames, is_video


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
    return meta


class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(self, meta_path, transform):
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)
        self.transform = transform

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        path = row["path"]

        if is_video(path):
            img = extract_frames(path, points=[0.5], backend="opencv")[0]
        else:
            img = pil_loader(path)

        img = self.transform(img)

        text = row["text"]
        text = clip.tokenize(text, truncate=True).squeeze()

        return img, text, index

    def __len__(self):
        return len(self.meta)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
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
    out_path = f"{wo_ext}_match{ext}"
    if args.skip_if_existing and os.path.exists(out_path):
        print(f"Output meta file '{out_path}' already exists. Exit.")
        exit()

    colossalai.launch_from_torch({})

    # build model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, preprocess = clip.load("ViT-L/14", device=device)
    logit_scale = model.logit_scale.exp().item()

    # build dataset
    dataset = VideoTextDataset(meta_path=meta_path, transform=preprocess)
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

    # compute scores
    indices_list = []
    scores_list = []
    model.eval()
    for imgs, text, indices in tqdm(dataloader, disable=dist.get_rank() != 0):
        imgs = imgs.to(device)
        text = text.to(device)

        with torch.no_grad():
            feat_img = model.encode_image(imgs)
            feat_text = model.encode_text(text)

        feat_img = F.normalize(feat_img, dim=1)
        feat_text = F.normalize(feat_text, dim=1)
        clip_scores = logit_scale * (feat_img * feat_text).sum(dim=1)
        clip_scores = clip_scores.cpu().tolist()
        indices_list.extend(indices)
        scores_list.extend(clip_scores)

    gathered_list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_list, (indices_list, scores_list))
    if dist.get_rank() == 0:
        meta_new = merge_scores(gathered_list, dataset.meta, column="match")
        meta_new.to_csv(out_path, index=False)
        print(f"New meta with matching scores saved to '{out_path}'.")


if __name__ == "__main__":
    main()
