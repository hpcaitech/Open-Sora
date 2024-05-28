import argparse
import json
import os
from ast import literal_eval

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
                full_results[key] = format(val[0]*100, ".2f")
    

    output_file_path = os.path.join(args.score_dir, "all_results.json")
    with open(output_file_path, "w") as outfile:
        json.dump(full_results, outfile, indent=4, sort_keys=True)
    print(f"results saved to: {output_file_path}")


