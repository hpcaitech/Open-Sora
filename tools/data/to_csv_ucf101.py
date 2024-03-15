import csv
import os


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


root = "~/data/ucf101"
split = "videos"

root = os.path.expanduser(root)
video_lists = get_filelist(os.path.join(root, split))
classes = [x.split("/")[-2] for x in video_lists]
classes = [split_by_capital(x) for x in classes]
samples = list(zip(video_lists, classes))

with open(f"preprocess/ucf101_{split}.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(samples)

print(f"Saved {len(samples)} samples to preprocess/ucf101_{split}.csv.")
