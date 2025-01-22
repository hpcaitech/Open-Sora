import argparse
import os
from datetime import timedelta

import clip
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets.folder import pil_loader
from tqdm import tqdm

from tools.datasets.utils import extract_frames, is_video


class VideoDataset(torch.utils.data.Dataset):
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
        img_size = img.size  # W, H

        img = self.transform(img)

        ret = dict(index=index, images=img, img_size=str(img_size))
        return ret

    def __len__(self):
        return len(self.meta)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--bs", type=int, default=128, help="Batch size")
    parser.add_argument("--thresh", type=float, default=0.98, help="similarity thresh")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        exit()

    wo_ext, ext = os.path.splitext(meta_path)
    out_path_dedup = f"{wo_ext}_dedup{ext}"
    out_path_dup = f"{wo_ext}_dup{ext}"

    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    # build model
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Note: do not use torch.device('cuda')!!!
    model, preprocess = clip.load("ViT-L/14", device=device)
    # preprocess: resize shorter size to 224 by keeping ar, then center crop

    # build dataset
    dataset = VideoDataset(meta_path=meta_path, transform=preprocess)
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

    # encode images and store the feature
    print("Begin to generate features")
    model.eval()
    feat_list = []  # store the feature
    indices_list = []  # store the indices
    img_size_list = []
    for batch in tqdm(dataloader, disable=dist.get_rank() != 0):
        indices = batch["index"]
        imgs = batch["images"].to(device, non_blocking=True)
        img_size = batch["img_size"]

        with torch.no_grad():
            feat_img = model.encode_image(imgs)
        feat_img = F.normalize(feat_img, dim=1)

        feat_list.append(feat_img)
        indices_list.extend(indices.tolist())
        img_size_list.extend(img_size)

    feats = torch.cat(feat_list, dim=0)

    # all_gather
    feats_gathered = [torch.zeros_like(feats, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(feats_gathered, feats)
    feats_all = torch.cat(feats_gathered)

    indices_gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(indices_gathered, indices_list)
    indices_all = np.array([x for sub in indices_gathered for x in sub])

    img_size_gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(img_size_gathered, img_size_list)
    img_size_all = [x for sub in img_size_gathered for x in sub]

    indices_unique, indices_unique_idx = np.unique(indices_all, return_index=True)
    feats_unique = feats_all[torch.from_numpy(indices_unique_idx)]
    img_size_unique = [img_size_all[x] for x in indices_unique_idx]

    if dist.get_rank() == 0:
        # get similarity scores
        non_zero_list = []
        sim_scores_list = []
        chunk = 1000
        for idx in tqdm(range(0, feats_unique.shape[0], chunk)):
            sim_mat = torch.matmul(feats_unique[idx : idx + chunk], feats_unique[idx:].T).cpu().numpy()
            sim_mat_upper = np.triu(sim_mat, k=1)
            non_zero_i = np.nonzero(sim_mat_upper >= args.thresh)
            sim_scores_i = sim_mat[non_zero_i]

            non_zero_np = np.stack(non_zero_i) + idx  # [2, N]
            non_zero_list.append(non_zero_np)
            sim_scores_list.append(sim_scores_i)

        non_zero_indices = np.concatenate(non_zero_list, axis=1)
        sim_scores = np.concatenate(sim_scores_list)
        dup_dict = {}
        for x, y, s in zip(non_zero_indices[0].tolist(), non_zero_indices[1].tolist(), sim_scores.tolist()):
            # only count pairs with same the same size
            if img_size_unique[x] != img_size_unique[y]:
                continue

            if y not in dup_dict:
                dup_dict[y] = (x, s)
            elif dup_dict[y][1] < s:
                dup_dict[y] = (x, s)

        dup_list = [(k, v) for k, v in dup_dict.items()]
        dup_list = sorted(dup_list, key=lambda x: x[1][1], reverse=True)
        dup_inds = [x[0] for x in dup_list]
        sim_inds = [x[1][0] for x in dup_list]
        sim_scores_dup = [x[1][1] for x in dup_list]

        remain_inds = sorted(list(set(indices_unique.tolist()) - set(dup_inds)))

        # save
        meta_unique = dataset.meta.iloc[remain_inds]
        meta_unique.to_csv(out_path_dedup, index=False)
        print(f"New meta without duplication saved to '{out_path_dedup}'.")

        meta_dup = dataset.meta.iloc[dup_inds].copy().reset_index(drop=True)
        path_dup = dataset.meta.iloc[sim_inds]["path"].copy().reset_index(drop=True)
        meta_dup["path_dup"] = path_dup
        meta_dup["sim"] = sim_scores_dup
        meta_dup.to_csv(out_path_dup, index=False)
        print(f"New meta with duplicated samples saved to '{out_path_dup}'.")


if __name__ == "__main__":
    main()
