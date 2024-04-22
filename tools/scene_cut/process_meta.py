"""
1. format_raw_meta()
    - only keep intact videos
    - add 'path' column (abs path)
2. create_meta_for_folder()
"""

import os

# os.chdir('../..')
print(f"Current working directory: {os.getcwd()}")

import argparse
import json
from functools import partial

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm
from utils_video import is_intact_video


def has_downloaded_success(json_path):
    if not os.path.exists(json_path):
        return False

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            if "success" not in data or isinstance(data["success"], bool) is False or data["success"] is False:
                return False
    except Exception:
        return False

    return True


def append_format_pandarallel(meta_path, folder_path, mode=".json"):
    def is_intact(row, mode=".json"):
        video_id = row["id"]
        # video_path = os.path.join(root_raw, f"data/{split}/{video_id}.mp4")
        video_path = os.path.join(folder_path, f"{video_id}.mp4")
        row["path"] = video_path
        if mode == ".mp4":
            if is_intact_video(video_path):
                return True, video_path
            return False, video_path
        elif mode == ".json":
            # json_path = os.path.join(root_raw, f"data/{split}/{video_id}.json")
            json_path = os.path.join(folder_path, f"{video_id}.json")
            if has_downloaded_success(json_path):
                return True, video_path
            return False, video_path
        elif mode is None:
            return True, video_path
        else:
            raise ValueError

    meta_dirpath = os.path.dirname(meta_path)
    meta_fname = os.path.basename(meta_path)
    wo_ext, ext = os.path.splitext(meta_fname)

    pandarallel.initialize(progress_bar=True)
    is_intact_partial = partial(is_intact, mode=mode)

    meta = pd.read_csv(meta_path)
    ret = meta.parallel_apply(is_intact_partial, axis=1)
    intact, paths = list(zip(*ret))

    meta["intact"] = intact
    meta["path"] = paths
    out_path = os.path.join(meta_dirpath, f"{wo_ext}_path_intact.csv")
    meta.to_csv(out_path, index=False)
    print(f"New meta (shape={meta.shape}) with intact info saved to '{out_path}'")

    # meta_format = meta[meta['intact']]
    meta_format = meta[np.array(intact)]
    meta_format.drop("intact", axis=1, inplace=True)
    out_path = os.path.join(meta_dirpath, f"{wo_ext}_path-filtered.csv")
    meta_format.to_csv(out_path, index=False)
    print(f"New meta (shape={meta_format.shape}) with format info saved to '{out_path}'")


def create_subset(meta_path):
    meta = pd.read_csv(meta_path)
    meta_subset = meta.iloc[:100]

    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_head-100{ext}"
    meta_subset.to_csv(out_path, index=False)
    print(f"New meta (shape={meta_subset.shape}) saved to '{out_path}'")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="append_path", required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--folder_path", type=str, required=True)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--num_workers", default=5, type=int)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    meta_path = args.meta_path
    task = args.task

    if task == "append_path":
        append_format_pandarallel(meta_path=meta_path, folder_path=args.folder_path, mode=args.mode)
    elif task == "create_subset":
        create_subset(meta_path=meta_path)
    else:
        raise ValueError


if __name__ == "__main__":
    main()
