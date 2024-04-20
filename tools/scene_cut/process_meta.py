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


def split_meta_csv(chunk_size=60000):
    """
    Split csv into multiple small csv in order
    """
    root = "./data/Panda-70M"
    # meta_name = 'meta/panda70m_training_full.csv'
    meta_name = "meta/panda70m_training_10m.csv"
    # meta_name = 'meta/training_10m/train_0.csv'
    meta_path = os.path.join(root, meta_name)

    df = pd.read_csv(meta_path)
    num_rows = len(df)

    # Split the DataFrame into smaller DataFrames
    for idx, i in enumerate(range(0, num_rows, chunk_size)):
        df_i = df.iloc[i : i + chunk_size]
        out_path = os.path.join(root, f"meta/train_{idx}.csv")
        df_i.to_csv(out_path, index=False)

    # If there are remaining rows
    if num_rows > chunk_size and num_rows % chunk_size != 0:
        df_last = df.iloc[-(num_rows % chunk_size) :]
        out_path = os.path.join(root, f"meta/train_{idx + 1}.csv")
        df_last.to_csv(out_path, index=False)


def remove_index():
    df = pd.read_csv("your_file.csv", index_col=0)
    df.to_csv("your_file_without_index.csv", index=False)


def append_format(meta_path, mode=".mp4"):
    """
    Append _format to csv file:
        - filter broken videos; only intact videos are kept
        - add column 'path'

    input csv should satisfy:
        - name should be: {split}.csv
        - contain column 'videoID'/'videoId'
    """
    # meta_path = os.path.join(root, f'raw/meta/{split}.csv')
    meta_dirname = os.path.dirname(meta_path)
    assert meta_dirname.endswith("raw/meta")
    root_raw = os.path.dirname(meta_dirname)

    meta_fname = os.path.basename(meta_path)
    split, ext = os.path.splitext(meta_fname)

    meta = pd.read_csv(meta_path)

    path_list = []
    new_meta = []
    for idx, row in tqdm(meta.iterrows(), total=len(meta)):
        # video_id = row['videoID']  # panda
        video_id = row["videoId"]  # pexels_new
        video_path = os.path.join(root_raw, f"data/{split}/{video_id}.mp4")
        if mode == ".mp4":
            if not is_intact_video(video_path):
                continue
        elif mode == ".json":
            json_path = os.path.join(root_raw, f"data/{split}/{video_id}.json")
            if not has_downloaded_success(json_path):
                continue
        else:
            raise ValueError

        new_meta.append(row)
        path_list.append(video_path)

    new_meta = pd.DataFrame(new_meta)
    new_meta["path"] = path_list

    out_path = os.path.join(root_raw, f"meta/{split}_format.csv")
    new_meta.to_csv(out_path, index=False)
    print(f"New meta (shape={new_meta.shape}) saved to '{out_path}'")


def append_format_pandarallel(meta_path, split, mode=".mp4"):
    """
    Append _format to csv file:
        - filter broken videos; only intact videos are kept
        - add column 'path'

    input csv should satisfy:
        - name should be: {split}.csv
        - contain column 'videoID'/'videoId'
    """
    # meta_path = os.path.join(root, f'raw/meta/{split}.csv')
    meta_dirname = os.path.dirname(meta_path)
    assert meta_dirname.endswith("raw/meta")
    root_raw = os.path.dirname(meta_dirname)

    meta_fname = os.path.basename(meta_path)
    wo_ext, ext = os.path.splitext(meta_fname)

    meta = pd.read_csv(meta_path)

    def is_intact(row, mode=".json"):
        video_id = row["videoId"]  # pexels_new
        video_path = os.path.join(root_raw, f"data/{split}/{video_id}.mp4")
        row["path"] = video_path
        if mode == ".mp4":
            if is_intact_video(video_path):
                return True, video_path
            return False, video_path
        elif mode == ".json":
            json_path = os.path.join(root_raw, f"data/{split}/{video_id}.json")
            if has_downloaded_success(json_path):
                return True, video_path
            return False, video_path
        else:
            raise ValueError

    pandarallel.initialize(progress_bar=True)
    is_intact_partial = partial(is_intact, mode=mode)
    ret = meta.parallel_apply(is_intact_partial, axis=1)

    intact, paths = list(zip(*ret))

    meta["intact"] = intact
    meta["path"] = paths
    out_path = os.path.join(root_raw, f"meta/{wo_ext}_intact.csv")
    meta.to_csv(out_path, index=False)
    print(f"New meta (shape={meta.shape}) with intact info saved to '{out_path}'")

    # meta_format = meta[meta['intact']]
    meta_format = meta[np.array(intact)]
    meta_format.drop("intact", axis=1, inplace=True)
    out_path = os.path.join(root_raw, f"meta/{wo_ext}_format.csv")
    meta_format.to_csv(out_path, index=False)
    print(f"New meta (shape={meta_format.shape}) with format info saved to '{out_path}'")


def create_subset(meta_path):
    meta = pd.read_csv(meta_path)
    meta_subset = meta.iloc[:100]

    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_head-100{ext}"
    meta_subset.to_csv(out_path, index=False)
    print(f"New meta (shape={meta_subset.shape}) saved to '{out_path}'")


def append_cut(root="./data/Panda-70M"):
    """
    Append _cut to csv file
    input csv should satisfy:
        - name_should be {split}_intact.csv
        - contain column 'timestamp': list of timestamp
    """
    split = "test"
    meta_path = os.path.join(root, f"processed/meta/{split}_intact.csv")

    wo_ext, ext = os.path.splitext(meta_path)
    suffix = "cut"
    out_path = f"{wo_ext}_{suffix}{ext}"

    meta = pd.read_csv(meta_path)

    new_meta = []
    for idx, row in tqdm(meta.iterrows(), total=len(meta)):
        video_id = row["videoID"]
        timestamps = eval(row["timestamp"])
        captions = eval(row["caption"])
        scores = eval(row["matching_score"])

        num_clips = len(timestamps)
        for idx_c in range(num_clips):
            path_i = os.path.join(root, f"processed/{split}/{video_id}_scene-{idx_c}.mp4")
            # if not is_intact_video(path_i):
            #     continue

            row_i = [f"{video_id}_scene-{idx_c}", path_i, timestamps[idx_c], captions[idx_c], scores[idx_c]]

            new_meta.append(row_i)

    columns = ["videoID", "path", "timestamp", "text", "match_official"]
    new_meta = pd.DataFrame(new_meta, columns=columns)

    new_meta.to_csv(out_path, index=False)
    print(f"New meta (shape={new_meta.shape}) saved to '{out_path}'")


def debug_meta_topk():
    meta_path = "F:/Panda-70M/meta/test_intact_cut_flow.csv"
    meta = pd.read_csv(meta_path)

    score_column = "flow_score"
    topk = meta.nlargest(10, columns=score_column)
    topk_s = meta.nsmallest(200, columns=score_column)

    [(row["path"], row["caption"], row[score_column]) for idx, row in topk.iterrows()]
    [(row["path"], row["caption"], row[score_column]) for idx, row in topk_s.iterrows()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="append_format")
    parser.add_argument("--meta_path", default="./data/pexels_new/raw/meta/popular_1.csv")
    parser.add_argument("--split", default="popular_5")
    parser.add_argument("--num_workers", default=5, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # split_meta_csv()

    args = parse_args()
    meta_path = args.meta_path
    task = args.task

    if task == "append_format":
        # append_format(meta_path=meta_path, mode='.mp4')
        append_format_pandarallel(meta_path=meta_path, split=args.split, mode=".json")
    elif task == "create_subset":
        create_subset(meta_path=meta_path)
    else:
        raise ValueError

    # append_cut(root=root)
    # append_score(root=root)
    # debug_meta_topk()
