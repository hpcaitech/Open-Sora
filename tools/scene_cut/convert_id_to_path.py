import argparse
import json
import os
from functools import partial

import cv2
import numpy as np
import pandas as pd
from mmengine.logging import print_log
from moviepy.editor import VideoFileClip
from pandarallel import pandarallel
from tqdm import tqdm

tqdm.pandas()


def is_intact_video(video_path, mode="moviepy", verbose=False, logger=None):
    if not os.path.exists(video_path):
        if verbose:
            print_log(f"Could not find '{video_path}'", logger=logger)
        return False

    if mode == "moviepy":
        try:
            VideoFileClip(video_path)
            if verbose:
                print_log(f"The video file '{video_path}' is intact.", logger=logger)
            return True
        except Exception as e:
            if verbose:
                print_log(f"Error: {e}", logger=logger)
                print_log(f"The video file '{video_path}' is not intact.", logger=logger)
            return False
    elif mode == "cv2":
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                if verbose:
                    print_log(f"The video file '{video_path}' is intact.", logger=logger)
                return True
        except Exception as e:
            if verbose:
                print_log(f"Error: {e}", logger=logger)
                print_log(f"The video file '{video_path}' is not intact.", logger=logger)
            return False
    else:
        raise ValueError


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str)
    parser.add_argument("--folder_path", type=str, required=True)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None, help="#workers for pandarallel")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    meta_path = args.meta_path
    folder_path = args.folder_path
    mode = args.mode

    def is_intact(row, mode=None):
        video_id = row["id"]
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

    if args.num_workers is not None:
        pandarallel.initialize(progress_bar=True, nb_workers=args.num_workers)
    else:
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

    meta_format = meta[np.array(intact)]
    meta_format.drop("intact", axis=1, inplace=True)
    out_path = os.path.join(meta_dirpath, f"{wo_ext}_path-filtered.csv")
    meta_format.to_csv(out_path, index=False)
    print(f"New meta (shape={meta_format.shape}) with format info saved to '{out_path}'")


if __name__ == "__main__":
    main()
