import argparse
import os
import random

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import IMG_EXTENSIONS, extract_frames

tqdm.pandas()

try:
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=True)
    pandas_has_parallel = True
except ImportError:
    pandas_has_parallel = False


def apply(df, func, **kwargs):
    if pandas_has_parallel:
        return df.parallel_apply(func, **kwargs)
    return df.progress_apply(func, **kwargs)


def get_new_path(path, input_dir, output):
    path_new = os.path.join(output, os.path.relpath(path, input_dir))
    os.makedirs(os.path.dirname(path_new), exist_ok=True)
    return path_new


def resize(path, length, input_dir, output):
    path_new = get_new_path(path, input_dir, output)
    ext = os.path.splitext(path)[1].lower()
    assert ext in IMG_EXTENSIONS
    img = cv2.imread(path)
    if img is not None:
        h, w = img.shape[:2]
        if min(h, w) > length:
            if h > w:
                new_h = length
                new_w = int(w * new_h / h)
            else:
                new_w = length
                new_h = int(h * new_w / w)
            img = cv2.resize(img, (new_w, new_h))
        cv2.imwrite(path_new, img)
    else:
        path_new = ""
    return path_new


def rand_crop(path, input_dir, output):
    ext = os.path.splitext(path)[1].lower()
    path_new = get_new_path(path, input_dir, output)
    assert ext in IMG_EXTENSIONS
    img = cv2.imread(path)
    if img is not None:
        h, w = img.shape[:2]
        width, height, _ = img.shape
        pos = random.randint(0, 3)
        if pos == 0:
            img_cropped = img[: width // 2, : height // 2]
        elif pos == 1:
            img_cropped = img[width // 2 :, : height // 2]
        elif pos == 2:
            img_cropped = img[: width // 2, height // 2 :]
        else:
            img_cropped = img[width // 2 :, height // 2 :]
        cv2.imwrite(path_new, img_cropped)
    else:
        path_new = ""
    return path_new


def main(args):
    data = pd.read_csv(args.input)
    if args.method == "img_rand_crop":
        data["path"] = apply(data["path"], lambda x: rand_crop(x, args.input_dir, args.output))
        output_csv = args.input.replace(".csv", f"_rand_crop.csv")
    elif args.method == "img_resize":
        data["path"] = apply(data["path"], lambda x: resize(x, args.length, args.input_dir, args.output))
        output_csv = args.input.replace(".csv", f"_resized{args.length}.csv")
    elif args.method == "vid_frame_extract":
        points = args.points if args.points is not None else args.points_index
        data = pd.DataFrame(np.repeat(data.values, 3, axis=0), columns=data.columns)
        num_points = len(points)
        data["point"] = np.nan
        for i, point in enumerate(points):
            if isinstance(point, int):
                data.loc[i::num_points, "point"] = point
            else:
                data.loc[i::num_points, "point"] = data.loc[i::num_points, "num_frames"] * point
        data["path"] = apply(data, lambda x: extract_frames(x["path"], args.input_dir, args.output, x["point"]), axis=1)
        output_csv = args.input.replace(".csv", f"_vid_frame_extract.csv")

    data.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str, choices=["img_resize", "img_rand_crop", "vid_frame_extract"])
    parser.add_argument("input", type=str)
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--disable-parallel", action="store_true")
    parser.add_argument("--length", type=int, default=2160)
    parser.add_argument("--seed", type=int, default=42, help="seed for random")
    parser.add_argument("--points", nargs="+", type=float, default=None)
    parser.add_argument("--points_index", nargs="+", type=int, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    if args.disable_parallel:
        pandas_has_parallel = False
    main(args)
