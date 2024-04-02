import argparse
import os

import cv2
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

try:
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=True)
    pandas_has_parallel = True
except ImportError:
    pandas_has_parallel = False


def apply(df, func):
    if pandas_has_parallel:
        return df.parallel_apply(func)
    return df.progress_apply(func)


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def get_new_path(path, input_dir, output):
    path_new = os.path.join(output, os.path.relpath(path, input_dir))
    os.makedirs(os.path.dirname(path_new), exist_ok=True)
    return path_new


def resize(path, length, input_dir, output):
    path_new = get_new_path(path, input_dir, output)
    ext = os.path.splitext(path)[1].lower()
    if ext in IMG_EXTENSIONS:
        img = cv2.imread(path)
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
        pass
    return path_new


def main(args):
    data = pd.read_csv(args.input)
    data["path"] = apply(data["path"], lambda x: resize(x, args.length, args.input_dir, args.output))
    output_csv = args.input.replace(".csv", f"_resized{args.length}.csv")
    data.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--disable-parallel", action="store_true")
    parser.add_argument("--length", type=int, default=2160)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.disable_parallel:
        pandas_has_parallel = False
    main(args)
