import argparse
import json

import pandas as pd
from pandarallel import pandarallel


def process_csv(original_csv_path, original_video_format="mp4"):
    pandarallel.initialize(progress_bar=True)

    # Construct the new CSV file name by adding '_caption' before the '.csv' extension
    caption_file = original_csv_path.replace(".csv", "_caption.csv")
    df = pd.read_csv(original_csv_path)

    # Add a new column for captions initialized with 'None'
    df["caption"] = "None"

    def process_row(row):
        path = row["path"]
        json_path = path.replace(original_video_format, "json")

        with open(json_path, "r") as f:
            json_data = json.load(f)

        row["caption"] = json_data["caption"]
        return row

    # Iterate over each row to replace video format with json in the path, and extract captions
    df = df.parallel_apply(process_row, axis=1)

    # Save the modified DataFrame to a new CSV file
    df.to_csv(caption_file, index=False)
    print(f"New CSV file with captions is saved as {caption_file}")


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Process a CSV file to add video captions.")
    parser.add_argument("csv_path", type=str, help="The path to the original CSV file.")
    parser.add_argument("video_format", nargs="?", default="mp4", help="The original video format (default: mp4).")

    # Parse arguments
    args = parser.parse_args()

    # Call the function with the provided arguments
    process_csv(args.csv_path, args.video_format)
