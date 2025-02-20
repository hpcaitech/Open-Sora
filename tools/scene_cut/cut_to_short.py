import argparse
import os
import subprocess
from functools import partial

import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str)
    parser.add_argument("--length", type=int, required=True, help="segment length in seconds; example: 1800")
    parser.add_argument("--src_dir", type=str, required=True, help="exmaple: /path/to/dataset/raw/")
    parser.add_argument("--dst_dir", type=str, required=True, help="exmaple: /path/to/dataset/cut_to_30min/")

    parser.add_argument("--num_workers", type=int, default=None, help="#workers for pandarallel")
    parser.add_argument("--disable_parallel", action="store_true", help="disable parallel processing")

    args = parser.parse_args()
    return args


def process_single_row(row, args):
    path = row["path"]
    assert path.startswith(args.src_dir), f"\npath: {path}\nsrc_dir:{args.src_dir}"

    out_path = os.path.join(args.dst_dir, os.path.relpath(path, args.src_dir))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wo_ext, ext = os.path.splitext(out_path)
    cmd = (
        f"ffmpeg -i {path} "
        f"-c copy -an "  # -an: no audio
        f"-f segment -segment_time 60 -reset_timestamps 1 -map 0 -segment_start_number 0 "
        f"{wo_ext}_%03d{ext}"
    )
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    stdout, stderr = proc.communicate()


def main():
    args = parse_args()
    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        exit()

    # initialize pandarallel
    tqdm.pandas()
    if not args.disable_parallel:
        if args.num_workers is not None:
            pandarallel.initialize(progress_bar=True, nb_workers=args.num_workers)
        else:
            pandarallel.initialize(progress_bar=True)
    process_single_row_partial = partial(process_single_row, args=args)

    # process
    meta = pd.read_csv(meta_path)
    if not args.disable_parallel:
        meta.parallel_apply(process_single_row_partial, axis=1)
    else:
        meta.apply(process_single_row_partial, axis=1)

    print("cut_to_short.py finished successfully.")


if __name__ == "__main__":
    main()
