# adapted from https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/simple_inference.py
import clip
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def get_video_length(cap):
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def extract_frames(video_path, points=(0.5,)):
    cap = cv2.VideoCapture(video_path)
    length = get_video_length(cap)
    points = [int(length * point) for point in points]
    frames = []
    if length < 3:
        return frames, length
    for point in points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, point)
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(frame)
    if len(frames) == 1:
        frames = frames[0]
    return frames, length


class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None):
        self.csv_path = csv_path
        self.samples = pd.read_csv(csv_path, header=None)
        self.transform = transform

    def getitem(self, index):
        sample = self.samples.iloc[index]
        img = extract_frames(sample[0])[0]
        img = self.transform(img)
        text = sample[1]

        return dict(index=index, image=img, text=text)

    def __len__(self):
        return len(self.samples)

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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AestheticScorer(768, device)

    dataset = VideoTextDataset(
        "/mnt/hdd/data/VidProM/VidProM_pika/meta/vidprom_relength_fmin_48_clean_en_unescape_nourl.csv",
        transform=model.preprocess,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=16, pin_memory=True)
    dataset.samples["aesthetic"] = ""
    model = torch.nn.DataParallel(model)
    output_file = "vidprom_aes.csv"
    index = 0
    for batch in tqdm(dataloader):
        image = batch["image"].to(device)
        with torch.no_grad():
            score = model(image)
        dataset.samples.loc[index : index + len(score) - 1, "aesthetic"] = score.cpu().numpy().flatten()
        index += len(score)

    dataset.samples.to_csv(output_file, index=False, header=False)
    print(f"Saved {index} samples")


if __name__ == "__main__":
    main()
