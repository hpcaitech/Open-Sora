import argparse
import os

import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

try:
    pass

    PANDA_USE_PARALLEL = True
except ImportError:
    PANDA_USE_PARALLEL = False


PANDA_USE_PARALLEL = False


def apply(df, func, **kwargs):
    if PANDA_USE_PARALLEL:
        return df.parallel_apply(func, **kwargs)
    return df.progress_apply(func, **kwargs)


def read_file(input_path):
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path)
    elif input_path.endswith(".parquet"):
        return dd.read_parquet(input_path).compute()
    else:
        raise NotImplementedError(f"Unsupported file format: {input_path}")


def save_file(df, output_path):
    if output_path.endswith(".csv"):
        df.to_csv(output_path, index=False)
    elif output_path.endswith(".parquet"):
        df.to_parquet(output_path, index=False)
    else:
        raise NotImplementedError(f"Unsupported file format: {output_path}")


def process_path(path, suffix):
    # Split path into parts
    parts = path.split("/")
    assert parts[4] == "data", f"Invalid path: {path}"
    parts[4] = "latents"

    # Process filename
    filename = parts[-1]
    ext = filename.split(".")[-1]
    filename = filename.replace(f".{ext}", suffix)
    parts[-1] = filename

    new_path = "/".join(parts)

    # Create directory if not exists
    directory = os.path.dirname(new_path)
    os.makedirs(directory, exist_ok=True)

    return new_path


def extend_csv(source_csv, target_csv):
    # Read CSV file
    df = read_file(source_csv)

    # Add new columns
    df["latents_path"] = apply(df["path"], lambda x: process_path(x, ".pt"))
    df["text_t5_path"] = apply(df["path"], lambda x: process_path(x, "_t5.pt"))
    df["text_clip_path"] = apply(df["path"], lambda x: process_path(x, "_clip.pt"))

    # Save modified CSV
    save_file(df, target_csv)
    print(f"CSV file saved to: {target_csv}")


def main():
    parser = argparse.ArgumentParser(description="Extend CSV file by adding latents and text encoding paths")
    parser.add_argument("--source_csv", required=True, help="Path to input CSV file")
    parser.add_argument("--target_csv", required=True, help="Path to output CSV file (optional)", default=None)

    args = parser.parse_args()
    extend_csv(args.source_csv, args.target_csv)


if __name__ == "__main__":
    main()
