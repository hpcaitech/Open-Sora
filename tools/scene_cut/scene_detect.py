import argparse
import os

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from scenedetect import AdaptiveDetector, detect
from tqdm import tqdm

tqdm.pandas()


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str)
    parser.add_argument("--num_workers", type=int, default=None, help="#workers for pandarallel")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        exit()

    if args.num_workers is not None:
        pandarallel.initialize(progress_bar=True, nb_workers=args.num_workers)
    else:
        pandarallel.initialize(progress_bar=True)

    meta = pd.read_csv(meta_path)
    ret = meta.parallel_apply(process_single_row, axis=1)

    succ, timestamps = list(zip(*ret))
    meta["timestamp"] = timestamps
    meta = meta[np.array(succ)]

    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_timestamp{ext}"
    meta.to_csv(out_path, index=False)
    print(f"New meta (shape={meta.shape}) with timestamp saved to '{out_path}'.")


if __name__ == "__main__":
    main()
