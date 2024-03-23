import argparse
import csv
import os

from torchvision.datasets import ImageNet


def get_filelist(file_path):
    Filelist = []
    for home, dirs, files in os.walk(file_path):
        for filename in files:
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

    with open(output, "w") as f:
        writer = csv.writer(f)
        writer.writerows(samples)

    print(f"Saved {len(samples)} samples to {output}.")


def process_ucf101(root, split):
    root = os.path.expanduser(root)
    video_lists = get_filelist(os.path.join(root, split))
    classes = [x.split("/")[-2] for x in video_lists]
    classes = [split_by_capital(x) for x in classes]
    samples = list(zip(video_lists, classes))
    output = f"ucf101_{split}.csv"

    with open(output, "w") as f:
        writer = csv.writer(f)
        writer.writerows(samples)

    print(f"Saved {len(samples)} samples to {output}.")


def process_vidprom(root, info):
    root = os.path.expanduser(root)
    video_lists = get_filelist(root)
    video_set = set(video_lists)
    # read info csv
    with open(info, "r") as f:
        reader = csv.reader(f)
        lines = list(reader)
    infos = [(os.path.join(root, f"pika-{x[0]}.mp4"), x[1]) for x in lines[1:]]
    samples = [x for x in infos if x[0] in video_set]
    output = "vidprom.csv"

    with open(output, "w") as f:
        writer = csv.writer(f)
        writer.writerows(samples)

    print(f"Saved {len(samples)} samples to {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=["imagenet", "ucf101", "vidprom"])
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
    else:
        raise ValueError("Invalid dataset")
