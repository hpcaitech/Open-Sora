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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    files = os.listdir(args.log_dir)
    files = [
        "img_0.log",
        "img_1.log",
        "img_2.log",
        "144p_vid.log",
        "240p_vid.log",
        "360p_vid.log",
        "480p_vid.log",
        "720p_vid.log",
    ]

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
    output_file_path = os.path.join(args.log_dir, "loss.json")
    with open(output_file_path, "w") as outfile:
        json.dump(loss_info, outfile, indent=4, sort_keys=True)
    print(f"results saved to: {output_file_path}")
