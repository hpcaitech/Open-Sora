import argparse
import json
import os
from ast import literal_eval

I2V_WEIGHT = 1.0
I2V_QUALITY_WEIGHT = 1.0

I2V_LIST = [
    "Video-Image Subject Consistency",
    "Video-Image Background Consistency",
]

I2V_QUALITY_LIST = [
    "Subject Consistency",
    "Background Consistency",
    "Motion Smoothness",
    "Dynamic Degree",
    "Aesthetic Quality",
    "Imaging Quality",
    "Temporal Flickering"
]

DIM_WEIGHT_I2V = {
"Video-Text Camera Motion": 0.1,
"Video-Image Subject Consistency": 1,
"Video-Image Background Consistency": 1,
"Subject Consistency": 1,
"Background Consistency": 1,
"Motion Smoothness": 1,
"Dynamic Degree": 0.5,
"Aesthetic Quality": 1,
"Imaging Quality": 1,
"Temporal Flickering": 1
}

NORMALIZE_DIC_I2V = {
    "Video-Text Camera Motion" :{"Min": 0.0, "Max":1.0 },
    "Video-Image Subject Consistency":{"Min": 0.1462, "Max": 1.0},
    "Video-Image Background Consistency":{"Min": 0.2615, "Max":1.0 },
    "Subject Consistency":{"Min": 0.1462, "Max": 1.0},
    "Background Consistency":{"Min": 0.2615, "Max": 1.0 },
    "Motion Smoothness":{"Min": 0.7060, "Max": 0.9975},
    "Dynamic Degree":{"Min": 0.0, "Max": 1.0},
    "Aesthetic Quality":{"Min": 0.0, "Max": 1.0},
    "Imaging Quality":{"Min": 0.0, "Max": 1.0},
    "Temporal Flickering":{"Min":0.6293, "Max": 1.0}
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_dir", type=str) # evaluation_results/samples_...
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
        scaled_score = (float(val) - NORMALIZE_DIC_I2V[dim]["Min"]) / (NORMALIZE_DIC_I2V[dim]["Max"] -  NORMALIZE_DIC_I2V[dim]["Min"])
        scaled_score *= DIM_WEIGHT_I2V[dim]
        scaled_results[dim] = scaled_score
        dims.add(dim)
    
    assert len(dims) == len(NORMALIZE_DIC_I2V), f"{set(NORMALIZE_DIC_I2V.keys())-dims} not calculated yet"

    quality_score = sum([scaled_results[i] for i in I2V_QUALITY_LIST]) / sum([DIM_WEIGHT_I2V[i] for i in I2V_QUALITY_LIST])
    i2v_score = sum([scaled_results[i] for i in I2V_LIST]) / sum([DIM_WEIGHT_I2V[i] for i in I2V_LIST])

    scaled_results["quality score"] = quality_score
    scaled_results["i2v score"] = i2v_score 
    scaled_results["total score"] = (quality_score * I2V_QUALITY_WEIGHT + i2v_score * I2V_WEIGHT) / (I2V_QUALITY_WEIGHT + I2V_WEIGHT)
    
    formated_scaled_results = {}
    for key,val in scaled_results.items():
        formated_scaled_results[key] = format(val*100, ".2f")+"%"

    output_file_path = os.path.join(args.score_dir, "all_results.json")
    with open(output_file_path, "w") as outfile:
        json.dump(full_results, outfile, indent=4, sort_keys=True)
    print(f"results saved to: {output_file_path}")


    scaled_file_path = os.path.join(args.score_dir, "scaled_results.json")
    with open(scaled_file_path, "w") as outfile:
        json.dump(formated_scaled_results, outfile, indent=4, sort_keys=True)
    print(f"results saved to: {scaled_file_path}")