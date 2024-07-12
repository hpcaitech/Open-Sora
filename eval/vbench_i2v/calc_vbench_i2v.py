import argparse
import os
import time

import torch
from vbench import VBench
from vbench2_beta_i2v import VBenchI2V

full_info_path = "eval/vbench_i2v/vbench2_i2v_full_info.json"
video_quality_dimensions = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "temporal_flickering",
]
i2v_dimensions = ["i2v_subject", "i2v_background", "camera_motion"]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_folder", type=str)  # samples/samples..._vbench_i2v/
    parser.add_argument("model_ckpt", type=str)
    parser.add_argument("--start", type=int, default=0)  # start index of dimension to be evaluated
    parser.add_argument("--end", type=int, default=-1)  # start index of dimension to be evaluated
    parser.add_argument("--calc_i2v", type=str2bool, default=True)
    parser.add_argument("--calc_quality", type=str2bool, default=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    output_dir = os.path.join(args.model_ckpt, "vbench_i2v")
    os.makedirs(output_dir, exist_ok=True)
    video_path = args.video_folder

    start_time = time.time()

    if args.calc_i2v:
        my_VBench_I2V = VBenchI2V(torch.device("cuda"), full_info_path, output_dir)
        end = args.end if args.end != -1 else len(i2v_dimensions)
        for i2v_dim in i2v_dimensions[args.start : end]:
            my_VBench_I2V.evaluate(videos_path=video_path, name=i2v_dim, dimension_list=[i2v_dim], resolution="1-1")

    kwargs = {}
    kwargs["imaging_quality_preprocessing_mode"] = "longer"  # use VBench/evaluate.py default

    if args.calc_quality:
        my_VBench = VBench(torch.device("cuda"), full_info_path, output_dir)
        end = args.end if args.end != -1 else len(video_quality_dimensions)
        for quality_dim in video_quality_dimensions[args.start : end]:
            my_VBench.evaluate(
                videos_path=video_path, name=quality_dim, dimension_list=[quality_dim], mode="vbench_standard", **kwargs
            )

    print("Runtime: %s seconds " % (time.time() - start_time))
