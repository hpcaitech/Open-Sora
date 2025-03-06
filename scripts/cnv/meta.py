import argparse

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from torchvision.io.video import read_video
from tqdm import tqdm


def set_parallel(num_workers: int = None) -> callable:
    if num_workers == 0:
        return lambda x, *args, **kwargs: x.progress_apply(*args, **kwargs)
    else:
        if num_workers is not None:
            pandarallel.initialize(progress_bar=True, nb_workers=num_workers)
        else:
            pandarallel.initialize(progress_bar=True)
        return lambda x, *args, **kwargs: x.parallel_apply(*args, **kwargs)


def get_video_info(path: str) -> pd.Series:
    vframes, _, vinfo = read_video(path, pts_unit="sec", output_format="TCHW")
    num_frames, C, height, width = vframes.shape
    fps = round(vinfo["video_fps"], 3)
    aspect_ratio = height / width if width > 0 else np.nan
    resolution = height * width

    ret = pd.Series(
        [height, width, fps, num_frames, aspect_ratio, resolution],
        index=[
            "height",
            "width",
            "fps",
            "num_frames",
            "aspect_ratio",
            "resolution",
        ],
        dtype=object,
    )
    return ret


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input file path")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument(
        "--num_workers", type=int, default=None, help="Number of workers"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input
    output_path = args.output
    num_workers = args.num_workers

    df = pd.read_csv(input_path)
    tqdm.pandas()
    apply = set_parallel(num_workers)

    result = apply(df["path"], get_video_info)
    for col in result.columns:
        df[col] = result[col]
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
