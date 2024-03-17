import csv
import os

from torchvision.datasets import ImageNet

root = "~/data/imagenet"
split = "train"

root = os.path.expanduser(root)
data = ImageNet(root, split=split)
samples = [(path, data.classes[label][0]) for path, label in data.samples]

with open(f"preprocess/imagenet_{split}.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(samples)

print(f"Saved {len(samples)} samples to preprocess/imagenet_{split}.csv.")
