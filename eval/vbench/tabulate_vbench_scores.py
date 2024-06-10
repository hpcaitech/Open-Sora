import argparse
import json
import os

SEMANTIC_WEIGHT = 1
QUALITY_WEIGHT = 4

QUALITY_LIST = [
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "aesthetic quality",
    "imaging quality",
    "dynamic degree",
]

SEMANTIC_LIST = [
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency",
]

NORMALIZE_DIC = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering": {"Min": 0.6293, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
    "object class": {"Min": 0.0, "Max": 1.0},
    "multiple objects": {"Min": 0.0, "Max": 1.0},
    "human action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0},
    "spatial relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222},
    "appearance style": {"Min": 0.0009, "Max": 0.2855},
    "temporal style": {"Min": 0.0, "Max": 0.364},
    "overall consistency": {"Min": 0.0, "Max": 0.364},
}

DIM_WEIGHT = {
    "subject consistency": 1,
    "background consistency": 1,
    "temporal flickering": 1,
    "motion smoothness": 1,
    "aesthetic quality": 1,
    "imaging quality": 1,
    "dynamic degree": 0.5,
    "object class": 1,
    "multiple objects": 1,
    "human action": 1,
    "color": 1,
    "spatial relationship": 1,
    "scene": 1,
    "appearance style": 1,
    "temporal style": 1,
    "overall consistency": 1,
}

ordered_scaled_res = [
    "total score",
    "quality score",
    "semantic score",
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "dynamic degree",
    "aesthetic quality",
    "imaging quality",
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_dir", type=str)  # ckpt_dir/eval/vbench
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
        dim = key.replace("_", " ") if "_" in key else key
        scaled_score = (float(val) - NORMALIZE_DIC[dim]["Min"]) / (
            NORMALIZE_DIC[dim]["Max"] - NORMALIZE_DIC[dim]["Min"]
        )
        scaled_score *= DIM_WEIGHT[dim]
        scaled_results[dim] = scaled_score
        dims.add(dim)

    assert len(dims) == len(NORMALIZE_DIC), f"{set(NORMALIZE_DIC.keys())-dims} not calculated yet"

    quality_score = sum([scaled_results[i] for i in QUALITY_LIST]) / sum([DIM_WEIGHT[i] for i in QUALITY_LIST])
    semantic_score = sum([scaled_results[i] for i in SEMANTIC_LIST]) / sum([DIM_WEIGHT[i] for i in SEMANTIC_LIST])
    scaled_results["quality score"] = quality_score
    scaled_results["semantic score"] = semantic_score
    scaled_results["total score"] = (quality_score * QUALITY_WEIGHT + semantic_score * SEMANTIC_WEIGHT) / (
        QUALITY_WEIGHT + SEMANTIC_WEIGHT
    )

    formated_scaled_results = {"items": []}
    for key in ordered_scaled_res:
        # formated_scaled_results[key] = format(val * 100, ".2f") + "%"
        formated_score = format(scaled_results[key] * 100, ".2f") + "%"
        formated_scaled_results["items"].append({key: formated_score})

    output_file_path = os.path.join(args.score_dir, "all_results.json")
    with open(output_file_path, "w") as outfile:
        json.dump(full_results, outfile, indent=4, sort_keys=True)
    print(f"results saved to: {output_file_path}")

    scaled_file_path = os.path.join(args.score_dir, "scaled_results.json")
    with open(scaled_file_path, "w") as outfile:
        json.dump(formated_scaled_results, outfile, indent=4, sort_keys=True)
    print(f"results saved to: {scaled_file_path}")
