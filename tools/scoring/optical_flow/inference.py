import os
# os.chdir('../..')
print(f'Current working directory: {os.getcwd()}')

import argparse
import av
import numpy as np
import pandas as pd
from einops import rearrange
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

import decord
from unimatch import UniMatch


def extract_frames_av(video_path, frame_inds=[0, 10, 20, 30]):
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    frames = []
    for idx in frame_inds:
        if idx >= total_frames:
            idx = total_frames - 1
        target_timestamp = int(idx * av.time_base / container.streams.video[0].average_rate)
        container.seek(target_timestamp)
        frame = next(container.decode(video=0)).to_image()
        frames.append(frame)
    return frames


def extract_frames(video_path, frame_inds=[0, 10, 20, 30]):
    container = decord.VideoReader(video_path, num_threads=1)
    total_frames = len(container)
    # avg_fps = container.get_avg_fps()

    frame_inds = np.array(frame_inds).astype(np.int32)
    frame_inds[frame_inds >= total_frames] = total_frames - 1
    frames = container.get_batch(frame_inds).asnumpy()  # [N, H, W, C]
    return frames


class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(self, meta_path, frame_inds=[0, 10, 20, 30]):
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)
        self.frame_inds = frame_inds

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        images = extract_frames(row["path"], frame_inds=self.frame_inds)
        # images = [pil_to_tensor(x) for x in images]  # [C, H, W]

        # transform
        images = torch.from_numpy(images).float()
        images = rearrange(images, 'N H W C -> N C H W')
        H, W = images.shape[-2:]
        if H > W:
            images = rearrange(images, 'N C H W -> N C W H')
        images = F.interpolate(images, size=(320, 576), mode='bilinear', align_corners=True)

        return images

    def __len__(self):
        return len(self.meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--bs", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
    args = parser.parse_args()

    meta_path = args.meta_path
    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f'{wo_ext}_flow{ext}'

    # build model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = UniMatch(
        feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=True,
        task='flow',
    )
    ckpt = torch.load(
        './pretrained_models/unimatch/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth'
    )
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    # build dataset
    dataset = VideoTextDataset(meta_path=meta_path, frame_inds=[0, 10, 20, 30])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # compute optical flow scores
    dataset.meta["flow"] = np.nan
    index = 0
    for images in tqdm(dataloader):
        images = images.to(device)
        B = images.shape[0]

        batch_0 = rearrange(images[:, :-1], 'B N C H W -> (B N) C H W').contiguous()
        batch_1 = rearrange(images[:, 1:], 'B N C H W -> (B N) C H W').contiguous()

        with torch.no_grad():
            res = model(
                batch_0, batch_1,
                attn_type='swin',
                attn_splits_list=[2, 8],
                corr_radius_list=[-1, 4],
                prop_radius_list=[-1, 1],
                num_reg_refine=6,
                task='flow',
                pred_bidir_flow=False,
            )
            flow_maps = res['flow_preds'][-1].cpu()  # [B * (N-1), 2, H, W]
            flow_maps = rearrange(flow_maps, '(B N) C H W -> B N H W C', B=B)
            flow_scores = flow_maps.abs().mean(dim=[1, 2, 3, 4])
            flow_scores_np = flow_scores.numpy()

        dataset.meta.loc[index: index + B - 1, "flow"] = flow_scores_np
        index += B

    dataset.meta.to_csv(out_path, index=False)
    print(f"New meta with optical flow scores saved to \'{out_path}\'.")


if __name__ == "__main__":
    main()
