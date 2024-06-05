import argparse
import os

from vbench import VBench
from vbench2_beta_i2v import VBenchI2V

FULL_INFO_PATH = "vbench2_beta_i2v/vbench2_i2v_full_info.json"
VIDEO_QUALITY_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "temporal_flickering",
]
I2V_DIMENSIONS = ["i2v_subject", "i2v_background", "camera_motion"]


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
    video_path = args.video_path

    my_VBench_I2V = VBenchI2V("cuda", FULL_INFO_PATH, "evaluation_results")
    my_VBench_I2V.evaluate(videos_path=video_path, name="vbench_i2v", dimension_list=I2V_DIMENSIONS, resolution="1-1")

    my_VBench = VBench("cuda", FULL_INFO_PATH, output_dir)
    my_VBench.evaluate(
        videos_path=video_path,
        name="vbench_video_quality",
        dimension_list=VIDEO_QUALITY_DIMENSIONS,
    )
