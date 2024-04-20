import argparse
import os

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from scenedetect import AdaptiveDetector, detect
from tqdm import tqdm


def process_single_row(row):
    # windows
    # from scenedetect import detect, ContentDetector, AdaptiveDetector

    video_path = row["path"]

    detector = AdaptiveDetector(
        adaptive_threshold=3.0,
        # luma_only=True,
    )
    # detector = ContentDetector()
    # TODO: catch error here
    try:
        scene_list = detect(video_path, detector, start_in_scene=True)
        timestamp = [(s.get_timecode(), t.get_timecode()) for s, t in scene_list]
        return True, str(timestamp)
    except Exception as e:
        print(f"Video '{video_path}' with error {e}")
        return False, ""


def main():
    meta_path = "F:/pexels_new/raw/meta/popular_1_format.csv"
    meta = pd.read_csv(meta_path)

    timestamp_list = []
    for idx, row in tqdm(meta.iterrows()):
        video_path = row["path"]

        detector = AdaptiveDetector(
            adaptive_threshold=1.5,
            luma_only=True,
        )
        # detector = ContentDetector()
        scene_list = detect(video_path, detector, start_in_scene=True)

        timestamp = [(s.get_timecode(), t.get_timecode()) for s, t in scene_list]
        timestamp_list.append(timestamp)

    meta["timestamp"] = timestamp_list

    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_timestamp{ext}"
    meta.to_csv(out_path, index=False)
    print(f"New meta with timestamp saved to '{out_path}'.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", default="F:/pexels_new/raw/meta/popular_1_format.csv")
    parser.add_argument("--num_workers", default=5, type=int)

    args = parser.parse_args()
    return args


def main_pandarallel():
    args = parse_args()
    meta_path = args.meta_path

    # meta_path = 'F:/pexels_new/raw/meta/popular_1_format.csv'
    meta = pd.read_csv(meta_path)

    pandarallel.initialize(progress_bar=True)
    ret = meta.parallel_apply(process_single_row, axis=1)

    succ, timestamps = list(zip(*ret))

    meta["timestamp"] = timestamps
    meta = meta[np.array(succ)]

    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_timestamp{ext}"
    meta.to_csv(out_path, index=False)
    print(f"New meta (shape={meta.shape}) with timestamp saved to '{out_path}'.")


if __name__ == "__main__":
    main_pandarallel()
