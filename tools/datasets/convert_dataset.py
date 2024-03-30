import argparse
import os

import pandas as pd
from torchvision.datasets import ImageNet

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def get_filelist(file_path, exts=None):
    Filelist = []
    for home, dirs, files in os.walk(file_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1].lower()
            if exts is None or ext in exts:
                Filelist.append(os.path.join(home, filename))
    return Filelist


def split_by_capital(name):
    # BoxingPunchingBag -> Boxing Punching Bag
    new_name = ""
    for i in range(len(name)):
        if name[i].isupper() and i != 0:
            new_name += " "
        new_name += name[i]
    return new_name


def process_imagenet(root, split):
    root = os.path.expanduser(root)
    data = ImageNet(root, split=split)
    samples = [(path, data.classes[label][0]) for path, label in data.samples]
    output = f"imagenet_{split}.csv"

    df = pd.DataFrame(samples, columns=["path", "text"])
    df.to_csv(output, index=False)
    print(f"Saved {len(samples)} samples to {output}.")


def process_ucf101(root, split):
    root = os.path.expanduser(root)
    video_lists = get_filelist(os.path.join(root, split))
    classes = [x.split("/")[-2] for x in video_lists]
    classes = [split_by_capital(x) for x in classes]
    samples = list(zip(video_lists, classes))
    output = f"ucf101_{split}.csv"

    df = pd.DataFrame(samples, columns=["path", "text"])
    df.to_csv(output, index=False)
    print(f"Saved {len(samples)} samples to {output}.")


def process_vidprom(root, info):
    root = os.path.expanduser(root)
    video_lists = get_filelist(root)
    video_set = set(video_lists)
    # read info csv
    infos = pd.read_csv(info)
    abs_path = infos["uuid"].apply(lambda x: os.path.join(root, f"pika-{x}.mp4"))
    is_exist = abs_path.apply(lambda x: x in video_set)
    df = pd.DataFrame(dict(path=abs_path[is_exist], text=infos["prompt"][is_exist]))
    df.to_csv("vidprom.csv", index=False)
    print(f"Saved {len(df)} samples to vidprom.csv.")


def process_general_images(root):
    root = os.path.expanduser(root)
    image_lists = get_filelist(root, IMG_EXTENSIONS)
    df = pd.DataFrame(dict(path=image_lists))
    df.to_csv("images.csv", index=False)
    print(f"Saved {len(df)} samples to images.csv.")


def process_general_videos(root):
    root = os.path.expanduser(root)
    video_lists = get_filelist(root, VID_EXTENSIONS)
    df = pd.DataFrame(dict(path=video_lists))
    df.to_csv("videos.csv", index=False)
    print(f"Saved {len(df)} samples to videos.csv.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=["imagenet", "ucf101", "vidprom", "image", "video"])
    parser.add_argument("root", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--info", type=str, default=None)
    args = parser.parse_args()

    if args.dataset == "imagenet":
        process_imagenet(args.root, args.split)
    elif args.dataset == "ucf101":
        process_ucf101(args.root, args.split)
    elif args.dataset == "vidprom":
        process_vidprom(args.root, args.info)
    elif args.dataset == "image":
        process_general_images(args.root)
    elif args.dataset == "video":
        process_general_videos(args.root)
    else:
        raise ValueError("Invalid dataset")
