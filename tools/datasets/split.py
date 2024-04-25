import argparse
from typing import List

import pandas as pd
from mmengine.config import Config

from opensora.datasets.bucket import Bucket


def split_by_bucket(
    bucket: Bucket,
    input_files: List[str],
    output_path: str,
    limit: int,
    frame_interval: int,
):
    print(f"Split {len(input_files)} files into {len(bucket)} buckets")
    total_limit = len(bucket) * limit
    bucket_cnt = {}
    # get all bucket id
    for hw_id, d in bucket.ar_criteria.items():
        for t_id, v in d.items():
            for ar_id in v.keys():
                bucket_id = (hw_id, t_id, ar_id)
                bucket_cnt[bucket_id] = 0
    output_df = None
    # split files
    for path in input_files:
        df = pd.read_csv(path)
        if output_df is None:
            output_df = pd.DataFrame(columns=df.columns)
        for i in range(len(df)):
            row = df.iloc[i]
            t, h, w = row["num_frames"], row["height"], row["width"]
            bucket_id = bucket.get_bucket_id(t, h, w, frame_interval)
            if bucket_id is None:
                continue
            if bucket_cnt[bucket_id] < limit:
                bucket_cnt[bucket_id] += 1
                output_df = pd.concat([output_df, pd.DataFrame([row])], ignore_index=True)
                if len(output_df) >= total_limit:
                    break
        if len(output_df) >= total_limit:
            break
    assert len(output_df) <= total_limit
    if len(output_df) == total_limit:
        print(f"All buckets are full ({total_limit} samples)")
    else:
        print(f"Only {len(output_df)} files are used")
    output_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, nargs="+")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-l", "--limit", default=200, type=int)
    args = parser.parse_args()
    assert args.limit > 0

    cfg = Config.fromfile(args.config)
    bucket_config = cfg.bucket_config
    # rewrite bucket_config
    for ar, d in bucket_config.items():
        for frames, t in d.items():
            p, bs = t
            if p > 0.0:
                p = 1.0
            d[frames] = (p, bs)
    bucket = Bucket(bucket_config)
    split_by_bucket(bucket, args.input, args.output, args.limit, cfg.dataset.frame_interval)
