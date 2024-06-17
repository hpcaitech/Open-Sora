import argparse
import os
import time

import torch

from vbench import VBench

full_info_path = "eval/vbench/VBench_full_info.json"
dimensions = [
    # Quality Score
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "temporal_flickering",
    # Semantic Score
    "object_class",
    "multiple_objects",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "overall_consistency",
    "human_action",
    "appearance_style",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_folder", type=str)  # samples/samples..._vbench/eval
    parser.add_argument("model_ckpt", type=str)
    parser.add_argument("--start", type=int, default=0)  # start index of dimension to be evaluated
    parser.add_argument("--end", type=int, default=-1)  # start index of dimension to be evaluated

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    output_dir = os.path.join(args.model_ckpt, "vbench")
    os.makedirs(output_dir, exist_ok=True)
    video_path = args.video_folder

    kwargs = {}
    kwargs["imaging_quality_preprocessing_mode"] = "longer"  # use VBench/evaluate.py default

    start_time = time.time()

    # NOTE: important to use torch.device("cuda"), else will have issue with object_class third_party module
    my_VBench = VBench(torch.device("cuda"), full_info_path, output_dir)
    if args.end == -1:  # adjust end accordingly
        args.end = len(dimensions)
    for dim in dimensions[args.start : args.end]:
        my_VBench.evaluate(
            videos_path=video_path,
            name=dim,
            local=False,
            read_frame=False,
            dimension_list=[dim],
            mode="vbench_standard",
            **kwargs,
        )

    print("Runtime: %s seconds " % (time.time() - start_time))
