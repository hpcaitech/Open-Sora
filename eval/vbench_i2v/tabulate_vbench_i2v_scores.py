import argparse
import json
import os

I2V_WEIGHT = 1.0
I2V_QUALITY_WEIGHT = 1.0

I2V_LIST = [
    "i2v_subject",
    "i2v_background",
]

I2V_QUALITY_LIST = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "dynamic_degree",
]

DIM_WEIGHT_I2V = {
    "camera_motion": 0.1,
    "i2v_subject": 1,
    "i2v_background": 1,
    "subject_consistency": 1,
    "background_consistency": 1,
    "motion_smoothness": 1,
    "dynamic_degree": 0.5,
    "aesthetic_quality": 1,
    "imaging_quality": 1,
    "temporal_flickering": 1,
}

NORMALIZE_DIC_I2V = {
    "camera_motion": {"Min": 0.0, "Max": 1.0},
    "i2v_subject": {"Min": 0.1462, "Max": 1.0},
    "i2v_background": {"Min": 0.2615, "Max": 1.0},
    "subject_consistency": {"Min": 0.1462, "Max": 1.0},
    "background_consistency": {"Min": 0.2615, "Max": 1.0},
    "motion_smoothness": {"Min": 0.7060, "Max": 0.9975},
    "dynamic_degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic_quality": {"Min": 0.0, "Max": 1.0},
    "imaging_quality": {"Min": 0.0, "Max": 1.0},
    "temporal_flickering": {"Min": 0.6293, "Max": 1.0},
}

ordered_scaled_res = [
    "total score",
    "i2v score",
    "quality score",
    "camera_motion",
    "i2v_subject",
    "i2v_background",
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "temporal_flickering",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_dir", type=str)  # ckpt_dir/eval/vbench_i2v
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    res_postfix = "_eval_results.json"
    info_postfix = "_full_info.json"
    files = os.listdir(args.score_dir)
    res_files = [x for x in files if res_postfix in x]
    info_files = [x for x in files if info_postfix in x]
    assert len(res_files) == len(info_files), f"got {len(res_files)} res files, but {len(info_files)} info files"

    full_results = {}
    for res_file in res_files:
        # first check if results is normal
        info_file = res_file.split(res_postfix)[0] + info_postfix
        with open(os.path.join(args.score_dir, info_file), "r", encoding="utf-8") as f:
            info = json.load(f)
            assert len(info[0]["video_list"]) > 0, f"Error: {info_file} has 0 video list"
        # read results
        with open(os.path.join(args.score_dir, res_file), "r", encoding="utf-8") as f:
            data = json.load(f)
            for key, val in data.items():
                full_results[key] = format(val[0], ".4f")

    scaled_results = {}
    dims = set()
    for key, val in full_results.items():
        dim = key
        scaled_score = (float(val) - NORMALIZE_DIC_I2V[dim]["Min"]) / (
            NORMALIZE_DIC_I2V[dim]["Max"] - NORMALIZE_DIC_I2V[dim]["Min"]
        )
        scaled_score *= DIM_WEIGHT_I2V[dim]
        scaled_results[dim] = scaled_score
        dims.add(dim)

    assert len(dims) == len(NORMALIZE_DIC_I2V), f"{set(NORMALIZE_DIC_I2V.keys())-dims} not calculated yet"

    quality_score = sum([scaled_results[i] for i in I2V_QUALITY_LIST]) / sum(
        [DIM_WEIGHT_I2V[i] for i in I2V_QUALITY_LIST]
    )
    i2v_score = sum([scaled_results[i] for i in I2V_LIST]) / sum([DIM_WEIGHT_I2V[i] for i in I2V_LIST])

    scaled_results["quality score"] = quality_score
    scaled_results["i2v score"] = i2v_score
    scaled_results["total score"] = (quality_score * I2V_QUALITY_WEIGHT + i2v_score * I2V_WEIGHT) / (
        I2V_QUALITY_WEIGHT + I2V_WEIGHT
    )

    formated_scaled_results = {"item": []}
    for key in ordered_scaled_res:
        formated_res = format(scaled_results[key] * 100, ".2f") + "%"
        formated_scaled_results["item"].append({key: formated_res})

    output_file_path = os.path.join(args.score_dir, "all_results.json")
    with open(output_file_path, "w") as outfile:
        json.dump(full_results, outfile, indent=4, sort_keys=True)
    print(f"results saved to: {output_file_path}")

    scaled_file_path = os.path.join(args.score_dir, "scaled_results.json")
    with open(scaled_file_path, "w") as outfile:
        json.dump(formated_scaled_results, outfile, indent=4, sort_keys=True)
    print(f"results saved to: {scaled_file_path}")
