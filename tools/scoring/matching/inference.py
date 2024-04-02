import argparse
import os

import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import clip


def extract_frames(video_path, points=[0.5]):
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    frames = []
    for point in points:
        target_frame = total_frames * point
        target_timestamp = int((target_frame * av.time_base) / container.streams.video[0].average_rate)
        container.seek(target_timestamp)
        frame = next(container.decode(video=0)).to_image()
        frames.append(frame)
    return frames


class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(self, meta_path, transform):
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)
        self.transform = transform

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        img = extract_frames(row["path"], points=[0.5])[0]
        img = self.transform(img)

        text = row['text']
        text = clip.tokenize(text).squeeze()

        return img, text

    def __len__(self):
        return len(self.meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
    args = parser.parse_args()

    meta_path = args.meta_path
    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_match{ext}"

    # build model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, preprocess = clip.load("ViT-L/14", device=device)
    logit_scale = model.logit_scale.exp().item()
    # model = torch.nn.DataParallel(model)

    # build dataset
    dataset = VideoTextDataset(meta_path=meta_path, transform=preprocess)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # compute scores
    dataset.meta["match"] = np.nan
    index = 0
    model.eval()
    for imgs, text in tqdm(dataloader):
        imgs = imgs.to(device)
        text = text.to(device)
        B = imgs.shape[0]

        with torch.no_grad():
            feat_img = model.encode_image(imgs)
            feat_text = model.encode_text(text)

        feat_img = F.normalize(feat_img, dim=1)
        feat_text = F.normalize(feat_text, dim=1)
        clip_scores = logit_scale * (feat_img * feat_text).sum(dim=1)
        clip_scores_np = clip_scores.cpu().numpy()

        dataset.meta.loc[index : index + B - 1, "match"] = clip_scores_np
        index += B

    dataset.meta.to_csv(out_path, index=False)
    print(f"New meta with matching scores saved to '{out_path}'.")


if __name__ == "__main__":
    main()
