import argparse
import os

import cv2
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

try:
    from pandarallel import pandarallel

    PANDA_USE_PARALLEL = True
except ImportError:
    PANDA_USE_PARALLEL = False


def save_first_frame(video_path, img_dir):
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return ""

    try:
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        if success:
            video_name = os.path.basename(video_path)
            image_name = os.path.splitext(video_name)[0] + "_first_frame.jpg"
            image_path = os.path.join(img_dir, image_name)

            cv2.imwrite(image_path, frame)
        else:
            raise ValueError("Video broken.")
        cap.release()
        return image_path
    except Exception as e:
        print(f"Save first frame of `{video_path}` failed. {e}")
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="path to the input csv dataset")
    parser.add_argument("--img-dir", type=str, help="path to save first frame image")
    parser.add_argument("--disable-parallel", action="store_true", help="disable parallel processing")
    parser.add_argument("--num-workers", type=int, default=None, help="number of workers")
    args = parser.parse_args()

    if args.disable_parallel:
        PANDA_USE_PARALLEL = False
    if PANDA_USE_PARALLEL:
        if args.num_workers is not None:
            pandarallel.initialize(nb_workers=args.num_workers, progress_bar=True)
        else:
            pandarallel.initialize(progress_bar=True)

    if not os.path.exists(args.img_dir):
        os.makedirs(args.img_dir)

    data = pd.read_csv(args.input)

    data["first_frame_path"] = data["path"].parallel_apply(save_first_frame, img_dir=args.img_dir)
    data_filtered = data.loc[data["first_frame_path"] != ""]
    output_csv_path = args.input.replace(".csv", "_first-frame.csv")
    data_filtered.to_csv(output_csv_path, index=False)
    print(f"First frame csv saved to: {output_csv_path}, first frame images saved to {args.img_dir}.")
