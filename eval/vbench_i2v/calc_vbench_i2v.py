import argparse
import os

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_folder", type=str)  # samples/samples..._vbench_i2v/
    parser.add_argument("model_ckpt", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    output_dir = os.path.join(args.model_ckpt, "eval", "vbench_i2v")
    os.makedirs(output_dir, exist_ok=True)
    video_path = args.video_folder

    my_VBench_I2V = VBenchI2V(torch.device("cuda"), full_info_path, output_dir)
    for i2v_dim in i2v_dimensions:
        my_VBench_I2V.evaluate(videos_path=video_path, name=i2v_dim, dimension_list=[i2v_dim], resolution="1-1")

    kwargs = {}
    kwargs["imaging_quality_preprocessing_mode"] = "longer"  # use VBench/evaluate.py default

    my_VBench = VBench(torch.device("cuda"), full_info_path, output_dir)
    for quality_dim in video_quality_dimensions:
        my_VBench.evaluate(
            videos_path=video_path, name=quality_dim, dimension_list=[quality_dim], mode="vbench_standard", **kwargs
        )
