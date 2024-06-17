import argparse
import os

import colossalai
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from mmengine import Config
from mmengine.dataset import Compose, default_collate
from mmengine.registry import DefaultScope
from mmocr.datasets import PackTextDetInputs
from mmocr.registry import MODELS
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import CenterCrop, Compose, Resize
from tqdm import tqdm

from tools.datasets.utils import extract_frames, is_video


def merge_scores(gathered_list: list, meta: pd.DataFrame):
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
    meta.loc[unique_indices, "ocr"] = flat_scores[unique_indices_idx]


class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(self, meta_path, transform):
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)
        self.transform = transform
        self.transform = Compose(
            [
                Resize(1024),
                CenterCrop(1024),
            ]
        )
        self.formatting = PackTextDetInputs(meta_keys=["scale_factor"])

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        path = row["path"]

        if is_video(path):
            img = extract_frames(path, frame_inds=[10], backend="opencv")[0]
        else:
            img = pil_loader(path)

        img = self.transform(img)
        img_array = np.array(img)[:, :, ::-1].copy()  # bgr
        results = {
            "img": img_array,
            "scale_factor": 1.0,
            # 'img_shape': img_array.shape[-2],
            # 'ori_shape': img_array.shape[-2],
        }
        results = self.formatting(results)
        results["index"] = index

        return results

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
    out_path = f"{wo_ext}_ocr{ext}"
    if args.skip_if_existing and os.path.exists(out_path):
        print(f"Output meta file '{out_path}' already exists. Exit.")
        exit()

    cfg = Config.fromfile("./tools/scoring/ocr/dbnetpp.py")
    colossalai.launch_from_torch({})

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    DefaultScope.get_instance("ocr", scope_name="mmocr")  # use mmocr Registry as default

    # build model
    model = MODELS.build(cfg.model)
    model.init_weights()
    model.to(device)  # set data_preprocessor._device
    print("==> Model built.")

    # build dataset
    transform = Compose(cfg.test_pipeline)
    dataset = VideoTextDataset(meta_path=meta_path, transform=transform)
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
        collate_fn=default_collate,
    )
    print("==> Dataloader built.")

    # compute scores
    dataset.meta["ocr"] = np.nan
    indices_list = []
    scores_list = []
    model.eval()
    for data in tqdm(dataloader, disable=dist.get_rank() != 0):
        indices_i = data["index"]
        indices_list.extend(indices_i.tolist())
        del data["index"]

        pred = model.test_step(data)  # this line will cast data to device

        num_texts_i = [(x.pred_instances.scores > 0.3).sum().item() for x in pred]
        scores_list.extend(num_texts_i)

    gathered_list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_list, (indices_list, scores_list))

    if dist.get_rank() == 0:
        merge_scores(gathered_list, dataset.meta)
        dataset.meta.to_csv(out_path, index=False)
        print(f"New meta (shape={dataset.meta.shape}) with ocr results saved to '{out_path}'.")


if __name__ == "__main__":
    main()
