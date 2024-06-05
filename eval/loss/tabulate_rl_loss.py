"""
usage:
    python tabulate_rl_loss.py --log_dir /home/zhengzangwei/projs/Open-Sora-dev/logs/loss --ckpt_name epoch0-global_step9000

save the processed json to:
    Open-Sora-dev/evaluation_results/rectified_flow/<ckpt_name>_loss.json
"""

import argparse
import json
import os
from ast import literal_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--ckpt_name", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    output_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.abspath(os.path.join(output_dir, "..", "..", "evaluation_results", "rectified_flow"))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = os.listdir(args.log_dir)
    files = ["img.log", "video_144p.log", "video_240p.log", "video_360p.log", "video_480p.log", "video_720p.log"]

    loss_info = {}

    for fname in files:
        path = os.path.join(args.log_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            content = f.readlines()
        eval_line = content[-1].split("losses:")[-1].strip()
        loss_dict = literal_eval(eval_line)
        for key, loss in loss_dict.items():
            resolution, frame = key
            if resolution not in loss_info:
                loss_info[resolution] = {}
            loss_info[resolution][frame] = format(loss, ".4f")

    # Convert and write JSON object to file
    output_file_path = os.path.join(output_dir, args.ckpt_name + "_loss.json")
    with open(output_file_path, "w") as outfile:
        json.dump(loss_info, outfile, indent=4, sort_keys=True)
    print(f"results saved to: {output_file_path}")
