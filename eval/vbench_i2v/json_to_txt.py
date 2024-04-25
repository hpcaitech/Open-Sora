import json
import os

RESOLUTIONS = ["1-1", "16-9", "7-4", "8-5"]

cache_root = "cache/crop"
resolution = RESOLUTIONS[0]
json_file = "vbench2_beta_i2v/vbench2_i2v_full_info.json"
save_path = "all_i2v.txt"

data = json.load(open(json_file))
txt = [
    f'{x["prompt_en"]}{{"reference_path": "{os.path.join(cache_root, resolution, x["image_name"])}", "mask_strategy": "0,0,0,1,0"}}'
    for x in data
]
with open(save_path, "w") as f:
    f.write("\n".join(txt))
