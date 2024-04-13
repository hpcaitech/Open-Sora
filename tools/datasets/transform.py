import argparse
import os
import random

import cv2
import decord
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

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
    assert ext in IMG_EXTENSIONS
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
    return path_new


def rand_crop(path, input_dir, output):
    ext = os.path.splitext(path)[1].lower()
    path_new = get_new_path(path, input_dir, output)
    assert ext in IMG_EXTENSIONS
    img = cv2.imread(path)
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
    return path_new


def extract_frames(video_path, input_dir, output, point):
    point = round(point)
    points = [point]
    path_new = get_new_path(video_path, input_dir, output)

    container = decord.VideoReader(video_path, num_threads=1)
    total_frames = len(container)
    frame_inds = np.array(points).astype(np.int32)
    frame_inds[frame_inds >= total_frames] = total_frames - 1
    frames = container.get_batch(frame_inds).asnumpy()
    frames_pil = Image.fromarray(frames[0])

    os.makedirs(path_new, exist_ok=True)
    path_new = os.path.join(path_new, f"{point}.jpg")
    frames_pil.save(path_new)
    return path_new


def extract_frames_new(
        video_path,
        frame_inds=None,
        points=None,
        backend='opencv',
        return_length=False,
):
    """
    Args:
        video_path (str): path to video
        frame_inds (List[int]): indices of frames to extract
        points (List[float]): values within [0, 1); multiply #frames to get frame indices
    Return:
        List[PIL.Image]
    """
    assert backend in ['av', 'opencv', 'decord']
    assert (frame_inds is None) or (points is None)

    if backend == 'av':
        import av
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames

        if points is not None:
            frame_inds = [int(p * total_frames) for p in points]

        frames = []
        for idx in frame_inds:
            if idx >= total_frames:
                idx = total_frames - 1
            target_timestamp = int(
                idx * av.time_base / container.streams.video[0].average_rate
            )
            container.seek(target_timestamp)
            frame = next(container.decode(video=0)).to_image()
            frames.append(frame)

        if return_length:
            return frames, total_frames
        return frames

    elif backend == 'decord':
        import decord
        container = decord.VideoReader(video_path, num_threads=1)
        total_frames = len(container)
        # avg_fps = container.get_avg_fps()

        if points is not None:
            frame_inds = [int(p * total_frames) for p in points]

        frame_inds = np.array(frame_inds).astype(np.int32)
        frame_inds[frame_inds >= total_frames] = total_frames - 1
        frames = container.get_batch(frame_inds).asnumpy()  # [N, H, W, C]
        frames = [Image.fromarray(x) for x in frames]

        if return_length:
            return frames, total_frames
        return frames

    elif backend == 'opencv':
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if points is not None:
            frame_inds = [int(p * total_frames) for p in points]

        frames = []
        for idx in frame_inds:
            if idx >= total_frames:
                idx = total_frames - 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)

        if return_length:
            return frames, total_frames
        return frames
    else:
        raise ValueError


def main(args):
    data = pd.read_csv(args.input)
    if args.method == "img_rand_crop":
        data["path"] = apply(data["path"], lambda x: rand_crop(x, args.input_dir, args.output))
    elif args.method == "img_resize":
        data["path"] = apply(data["path"], lambda x: resize(x, args.length, args.input_dir, args.output))
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

    output_csv = args.input.replace(".csv", f"_resized{args.length}.csv")
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
    exit()

    from torchvision.transforms.functional import pil_to_tensor
    ret = extract_frames_new(
        'E:/data/video/pexels_new/8974385_scene-0.mp4',
        frame_inds=[0, 50, 100, 150],
        backend='opencv')
    for idx, img in enumerate(ret):
        save_path = f'./checkpoints/vis/{idx}.png'
        ret[idx].save(save_path)
    exit()
