# adapted from https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/simple_inference.py
import argparse
import os

import av
import clip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from tqdm import tqdm

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def is_video(filename):
    ext = os.path.splitext(filename)[-1].lower()
    return ext in VID_EXTENSIONS


def extract_frames(video_path, points=(0.1, 0.5, 0.9)):
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


def get_image(image_path):
    return Image.open(image_path).convert("RGB")


class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None, points=(0.1, 0.5, 0.9)):
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.points = points

    def getitem(self, index):
        sample = self.data.iloc[index]
        path = sample["path"]
        if not is_video(path):
            images = [get_image(path)]
        else:
            images = extract_frames(sample["path"], points=self.points)
        images = [self.transform(img) for img in images]
        images = torch.stack(images)

        return dict(index=index, image=images)

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
    output_file = args.input.replace(".csv", "_aes.csv")

    # build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AestheticScorer(768, device)
    preprocess = model.preprocess
    model = torch.nn.DataParallel(model)

    # build dataset
    dataset = VideoTextDataset(args.input, transform=preprocess, points=(0.5,))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )

    # compute aesthetic scores
    dataset.data["aesthetic"] = np.nan
    index = 0
    for batch in tqdm(dataloader):
        images = batch["image"].to(device)
        B = images.shape[0]
        images = rearrange(images, "b p c h w -> (b p) c h w")
        with torch.no_grad():
            scores = model(images)
        scores = rearrange(scores, "(b p) 1 -> b p", b=B)
        scores = scores.mean(dim=1)
        scores_np = scores.cpu().numpy()
        dataset.data.loc[index : index + len(scores_np) - 1, "aesthetic"] = scores_np
        index += len(images)
    dataset.data.to_csv(output_file, index=False)
    print(f"Saved aesthetic scores to {output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to the input CSV file")
    parser.add_argument("--bs", type=int, default=512, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=64, help="Number of workers")
    parser.add_argument("--prefetch_factor", type=int, default=8, help="Prefetch factor")
    args = parser.parse_args()

    main(args)
