import argparse
import os
import random
import shutil
import subprocess

import cv2
import ffmpeg
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

from .utils import IMG_EXTENSIONS, extract_frames

tqdm.pandas()
USE_PANDARALLEL = True


def apply(df, func, **kwargs):
    if USE_PANDARALLEL:
        return df.parallel_apply(func, **kwargs)
    return df.progress_apply(func, **kwargs)


def get_new_path(path, input_dir, output):
    path_new = os.path.join(output, os.path.relpath(path, input_dir))
    os.makedirs(os.path.dirname(path_new), exist_ok=True)
    return path_new


def resize_longer(path, length, input_dir, output_dir):
    path_new = get_new_path(path, input_dir, output_dir)
    ext = os.path.splitext(path)[1].lower()
    assert ext in IMG_EXTENSIONS
    img = cv2.imread(path)
    if img is not None:
        h, w = img.shape[:2]
        if min(h, w) > length:
            if h > w:
                new_h = length
                new_w = int(w / h * length)
            else:
                new_w = length
                new_h = int(h / w * length)
            img = cv2.resize(img, (new_w, new_h))
        cv2.imwrite(path_new, img)
    else:
        path_new = ""
    return path_new


def resize_shorter(path, length, input_dir, output_dir):
    path_new = get_new_path(path, input_dir, output_dir)
    if os.path.exists(path_new):
        return path_new

    ext = os.path.splitext(path)[1].lower()
    assert ext in IMG_EXTENSIONS
    img = cv2.imread(path)
    if img is not None:
        h, w = img.shape[:2]
        if min(h, w) > length:
            if h > w:
                new_w = length
                new_h = int(h / w * length)
            else:
                new_h = length
                new_w = int(w / h * length)
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


def m2ts_to_mp4(row, output_dir):
    input_path = row["path"]
    output_name = os.path.basename(input_path).replace(".m2ts", ".mp4")
    output_path = os.path.join(output_dir, output_name)
    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        ffmpeg.input(input_path).output(output_path).overwrite_output().global_args("-loglevel", "quiet").run(
            capture_stdout=True
        )
        row["path"] = output_path
        row["relpath"] = os.path.splitext(row["relpath"])[0] + ".mp4"
    except Exception as e:
        print(f"Error converting {input_path} to mp4: {e}")
        row["path"] = ""
        row["relpath"] = ""
        return row
    return row


def mkv_to_mp4(row, output_dir):
    # str_to_replace and str_to_replace_with account for the different directory structure
    input_path = row["path"]
    output_name = os.path.basename(input_path).replace(".mkv", ".mp4")
    output_path = os.path.join(output_dir, output_name)

    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        ffmpeg.input(input_path).output(output_path).overwrite_output().global_args("-loglevel", "quiet").run(
            capture_stdout=True
        )
        row["path"] = output_path
        row["relpath"] = os.path.splitext(row["relpath"])[0] + ".mp4"
    except Exception as e:
        print(f"Error converting {input_path} to mp4: {e}")
        row["path"] = ""
        row["relpath"] = ""
        return row
    return row


def mp4_to_mp4(row, output_dir):
    # str_to_replace and str_to_replace_with account for the different directory structure
    input_path = row["path"]

    # 检查输入文件是否为.mp4文件
    if not input_path.lower().endswith(".mp4"):
        print(f"Error: {input_path} is not an .mp4 file.")
        row["path"] = ""
        row["relpath"] = ""
        return row
    output_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, output_name)

    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        shutil.copy2(input_path, output_path)  # 使用shutil复制文件
        row["path"] = output_path
        row["relpath"] = os.path.splitext(row["relpath"])[0] + ".mp4"
    except Exception as e:
        print(f"Error coy {input_path} to mp4: {e}")
        row["path"] = ""
        row["relpath"] = ""
        return row
    return row


def crop_to_square(input_path, output_path):
    cmd = (
        f"ffmpeg -i {input_path} "
        f"-vf \"crop='min(in_w,in_h)':'min(in_w,in_h)':'(in_w-min(in_w,in_h))/2':'(in_h-min(in_w,in_h))/2'\" "
        f"-c:v libx264 -an "
        f"-map 0:v {output_path}"
    )
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    stdout, stderr = proc.communicate()


def vid_crop_center(row, input_dir, output_dir):
    input_path = row["path"]
    relpath = os.path.relpath(input_path, input_dir)
    assert not relpath.startswith("..")
    output_path = os.path.join(output_dir, relpath)

    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        crop_to_square(input_path, output_path)
        size = min(row["height"], row["width"])
        row["path"] = output_path
        row["height"] = size
        row["width"] = size
        row["aspect_ratio"] = 1.0
        row["resolution"] = size**2
    except Exception as e:
        print(f"Error cropping {input_path} to center: {e}")
        row["path"] = ""
    return row


def main():
    args = parse_args()
    global USE_PANDARALLEL

    assert args.num_workers is None or not args.disable_parallel
    if args.disable_parallel:
        USE_PANDARALLEL = False
    if args.num_workers is not None:
        pandarallel.initialize(progress_bar=True, nb_workers=args.num_workers)
    else:
        pandarallel.initialize(progress_bar=True)

    random.seed(args.seed)
    data = pd.read_csv(args.meta_path)
    if args.task == "img_rand_crop":
        data["path"] = apply(data["path"], lambda x: rand_crop(x, args.input_dir, args.output_dir))
        output_csv = args.meta_path.replace(".csv", "_rand_crop.csv")
    elif args.task == "img_resize_longer":
        data["path"] = apply(data["path"], lambda x: resize_longer(x, args.length, args.input_dir, args.output_dir))
        output_csv = args.meta_path.replace(".csv", f"_resize-longer-{args.length}.csv")
    elif args.task == "img_resize_shorter":
        data["path"] = apply(data["path"], lambda x: resize_shorter(x, args.length, args.input_dir, args.output_dir))
        output_csv = args.meta_path.replace(".csv", f"_resize-shorter-{args.length}.csv")
    elif args.task == "vid_frame_extract":
        points = args.points if args.points is not None else args.points_index
        data = pd.DataFrame(np.repeat(data.values, 3, axis=0), columns=data.columns)
        num_points = len(points)
        data["point"] = np.nan
        for i, point in enumerate(points):
            if isinstance(point, int):
                data.loc[i::num_points, "point"] = point
            else:
                data.loc[i::num_points, "point"] = data.loc[i::num_points, "num_frames"] * point
        data["path"] = apply(
            data, lambda x: extract_frames(x["path"], args.input_dir, args.output_dir, x["point"]), axis=1
        )
        output_csv = args.meta_path.replace(".csv", "_vid_frame_extract.csv")
    elif args.task == "m2ts_to_mp4":
        print(f"m2ts_to_mp4作业开始：{args.output_dir}")
        assert args.meta_path.endswith("_m2ts.csv"), "Input file must end with '_m2ts.csv'"
        m2ts_to_mp4_partial = lambda x: m2ts_to_mp4(x, args.output_dir)
        data = apply(data, m2ts_to_mp4_partial, axis=1)
        data = data[data["path"] != ""]
        output_csv = args.meta_path.replace("_m2ts.csv", ".csv")
    elif args.task == "mkv_to_mp4":
        print(f"mkv_to_mp4作业开始：{args.output_dir}")
        assert args.meta_path.endswith("_mkv.csv"), "Input file must end with '_mkv.csv'"
        mkv_to_mp4_partial = lambda x: mkv_to_mp4(x, args.output_dir)
        data = apply(data, mkv_to_mp4_partial, axis=1)
        data = data[data["path"] != ""]
        output_csv = args.meta_path.replace("_mkv.csv", ".csv")
    elif args.task == "mp4_to_mp4":
        # assert args.meta_path.endswith("meta.csv"), "Input file must end with '_mkv.csv'"
        print(f"MP4复制作业开始：{args.output_dir}")
        mkv_to_mp4_partial = lambda x: mp4_to_mp4(x, args.output_dir)
        data = apply(data, mkv_to_mp4_partial, axis=1)
        data = data[data["path"] != ""]
        output_csv = args.meta_path
    elif args.task == "vid_crop_center":
        vid_crop_center_partial = lambda x: vid_crop_center(x, args.input_dir, args.output_dir)
        data = apply(data, vid_crop_center_partial, axis=1)
        data = data[data["path"] != ""]
        output_csv = args.meta_path.replace(".csv", "_center-crop.csv")
    else:
        raise ValueError
    data.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")
    raise SystemExit(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "img_resize_longer",
            "img_resize_shorter",
            "img_rand_crop",
            "vid_frame_extract",
            "m2ts_to_mp4",
            "mkv_to_mp4",
            "mp4_to_mp4",
            "vid_crop_center",
        ],
    )
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--length", type=int, default=1080)
    parser.add_argument("--disable-parallel", action="store_true")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="seed for random")
    parser.add_argument("--points", nargs="+", type=float, default=None)
    parser.add_argument("--points_index", nargs="+", type=int, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
