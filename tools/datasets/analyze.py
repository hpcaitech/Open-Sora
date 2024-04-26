import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def read_file(input_path):
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path)
    elif input_path.endswith(".parquet"):
        return pd.read_parquet(input_path)
    else:
        raise NotImplementedError(f"Unsupported file format: {input_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to the input dataset")
    parser.add_argument("--save-img", type=str, default="samples/infos/", help="Path to save the image")
    return parser.parse_args()


def plot_data(data, column, bins, name):
    plt.clf()
    data.hist(column=column, bins=bins)
    os.makedirs(os.path.dirname(name), exist_ok=True)
    plt.savefig(name)
    print(f"Saved {name}")


def plot_categorical_data(data, column, name):
    plt.clf()
    data[column].value_counts().plot(kind="bar")
    os.makedirs(os.path.dirname(name), exist_ok=True)
    plt.savefig(name)
    print(f"Saved {name}")


COLUMNS = {
    "num_frames": 100,
    "resolution": 100,
    "text_len": 100,
    "aes": 100,
    "match": 100,
    "flow": 100,
    "cmotion": None,
}


def main(args):
    data = read_file(args.input)

    # === Image Data Info ===
    image_index = data["num_frames"] == 1
    if image_index.sum() > 0:
        print("=== Image Data Info ===")
        img_data = data[image_index]
        print(f"Number of images: {len(img_data)}")
        print(img_data.head())
        print(img_data.describe())
        if args.save_img:
            for column in COLUMNS:
                if column in img_data.columns and column not in ["num_frames", "cmotion"]:
                    if COLUMNS[column] is None:
                        plot_categorical_data(img_data, column, os.path.join(args.save_img, f"image_{column}.png"))
                    else:
                        plot_data(img_data, column, COLUMNS[column], os.path.join(args.save_img, f"image_{column}.png"))

    # === Video Data Info ===
    if not image_index.all():
        print("=== Video Data Info ===")
        video_data = data[~image_index]
        print(f"Number of videos: {len(video_data)}")
        if "num_frames" in video_data.columns:
            total_num_frames = video_data["num_frames"].sum()
            print(f"Number of frames: {total_num_frames}")
            DEFAULT_FPS = 30
            total_hours = total_num_frames / DEFAULT_FPS / 3600
            print(f"Total hours (30 FPS): {int(total_hours)}")
        print(video_data.head())
        print(video_data.describe())
        if args.save_img:
            for column in COLUMNS:
                if column in video_data.columns:
                    if COLUMNS[column] is None:
                        plot_categorical_data(video_data, column, os.path.join(args.save_img, f"video_{column}.png"))
                    else:
                        plot_data(
                            video_data, column, COLUMNS[column], os.path.join(args.save_img, f"video_{column}.png")
                        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
