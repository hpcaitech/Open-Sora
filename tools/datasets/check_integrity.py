import argparse
import subprocess

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

try:
    from pandarallel import pandarallel

    PANDA_USE_PARALLEL = True
except ImportError:
    PANDA_USE_PARALLEL = False

import shutil

if not shutil.which("ffmpeg"):
    raise ImportError("FFmpeg is not installed")


def apply(df, func, **kwargs):
    if PANDA_USE_PARALLEL:
        return df.parallel_apply(func, **kwargs)
    return df.progress_apply(func, **kwargs)


def check_video_integrity(video_path):
    # try:
    can_open_result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", video_path, "-t", "0", "-f", "null", "-"],  # open video and capture 0 seconds
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    fast_scan_result = subprocess.run(
        ["ffmpeg", "-v", "error", "-analyzeduration", "10M", "-probesize", "10M", "-i", video_path, "-f", "null", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if can_open_result.stderr == "" and fast_scan_result.stderr == "":
        return True
    else:
        return False
    # except Exception as e:
    #     return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="path to the input dataset")
    parser.add_argument("--disable-parallel", action="store_true", help="disable parallel processing")
    parser.add_argument("--num-workers", type=int, default=None, help="number of workers")
    args = parser.parse_args()

    if args.disable_parallel:
        PANDA_USE_PARALLEL = False
    if PANDA_USE_PARALLEL:
        if args.num_workers is not None:
            pandarallel.initialize(nb_workers=args.num_workers, progress_bar=True)
        else:
            pandarallel.initialize(progress_bar=True)

    data = pd.read_csv(args.input)
    assert "path" in data.columns
    data["integrity"] = apply(data["path"], check_video_integrity)

    integrity_file_path = args.input.replace(".csv", "_intact.csv")
    broken_file_path = args.input.replace(".csv", "_broken.csv")

    intact_data = data[data["integrity"] == True].drop(columns=["integrity"])
    intact_data.to_csv(integrity_file_path, index=False)
    broken_data = data[data["integrity"] == False].drop(columns=["integrity"])
    broken_data.to_csv(broken_file_path, index=False)

    print(
        f"Integrity check completed. Intact videos saved to: {integrity_file_path}, broken videos saved to {broken_file_path}."
    )
