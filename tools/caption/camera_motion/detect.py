# Originally developed by https://github.com/Vchitect/VBench based on https://github.com/facebookresearch/co-tracker.

import argparse
from typing import List

import pandas as pd

from .camera_motion import compute_camera_motion


def process(paths: List[str], threshold: float) -> List[str]:
    device = "cuda"
    submodules = {"repo": "facebookresearch/co-tracker", "model": "cotracker2"}
    camera_motion_types = compute_camera_motion(device, submodules, paths, factor=threshold)
    return camera_motion_types


def main(args):
    output_file = args.input.replace(".csv", "_cmotion.csv")
    data = pd.read_csv(args.input)
    data["cmotion"] = process(data["path"], args.threshold)
    data.to_csv(output_file, index=False)
    print(f"Output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()
    main(args)
