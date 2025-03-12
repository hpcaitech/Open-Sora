import os

import pandas as pd
from tqdm import tqdm

try:
    import dask.dataframe as dd

    SUPPORT_DASK = True
except:
    SUPPORT_DASK = False


def shard_parquet(input_path, k):
    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file {input_path} does not exist.")

    # 读取 Parquet 文件为 Pandas DataFrame
    if SUPPORT_DASK:
        df = dd.read_parquet(input_path).compute()
    else:
        df = pd.read_parquet(input_path)

    # 去除指定的列
    columns_to_remove = [
        "num_frames",
        "height",
        "width",
        "aspect_ratio",
        "fps",
        "resolution",
    ]
    df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

    # 计算每个分片的大小
    total_rows = len(df)
    rows_per_shard = (total_rows + k - 1) // k  # 向上取整

    # 创建与原始文件同名的文件夹
    base_dir = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.join(base_dir, base_name)
    os.makedirs(output_dir, exist_ok=True)

    # 创建分片并保存到文件夹
    for i in tqdm(range(k)):
        start_idx = i * rows_per_shard
        end_idx = min(start_idx + rows_per_shard, total_rows)

        shard_df = df.iloc[start_idx:end_idx]
        if shard_df.empty:
            continue

        shard_file_name = f"{i + 1:05d}.parquet"
        shard_path = os.path.join(output_dir, shard_file_name)

        shard_df.to_parquet(shard_path, index=False)

        # print(f"Shard saved to {shard_path}, rows: {len(shard_df)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Shard a Parquet file.")
    parser.add_argument("input_path", type=str, help="Path to the input Parquet file.")
    parser.add_argument(
        "k", type=int, help="Number of shards to create.", default=10000
    )

    args = parser.parse_args()

    shard_parquet(args.input_path, args.k)
