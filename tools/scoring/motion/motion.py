# Please follow the instructions in the README.md file to install dependency.

import argparse
import json
import math
import multiprocessing
import os
import shutil
import subprocess

import pandas as pd
import tqdm

# Constants
TEMP_FOLDER = "/mnt/ddn/yeanbang/experiments/data_pipeline/temp_videos"
DOCKER_IMAGE = "vmaf_im"
EXTRACT_MVS_RUNNABLE_PATH = "~/extract_mvs"
GET_VMAF_MOTION_SCORE = True
GET_MOTION_VECTOR_AVERAGE = True

VMAF_CMD = """ffmpeg \
    -nostats -loglevel 0 \
    -r 24 -i "$SRC_VIDEO$" \
    -r 24 -i "$SRC_VIDEO$" \
    -lavfi "[0:v]setpts=PTS-STARTPTS[reference]; \
            [1:v]setpts=PTS-STARTPTS[distorted]; \
            [distorted][reference]libvmaf=log_fmt=json:log_path=$OUTPUT_PATH$:n_threads=4" \
    -f null -"""


def create_temp_folder():
    """Creates a temp folder if it doesn't exist."""
    if not os.path.exists(TEMP_FOLDER):
        os.makedirs(TEMP_FOLDER)


def calculate_magnitude(mvx, mvy):
    return math.sqrt(mvx**2 + mvy**2)


def get_average_motion_vector_strength(motion_vector_log_file):
    """Calculates the average strength of motion vectors in a log file."""

    # Initialize variables to calculate the average
    total_magnitude = 0
    vector_count = 0
    # Open and parse the log file
    df = pd.read_csv(motion_vector_log_file)
    motion_x = df["motion_x"]
    motion_y = df["motion_y"]
    # Calculate the magnitude of each motion vector
    for i in range(len(motion_x)):
        magnitude = calculate_magnitude(motion_x[i], motion_y[i])
        total_magnitude += magnitude
        vector_count += 1
    if vector_count == 0:
        return 0
    return total_magnitude / vector_count


def process_video(video_info):
    """Processes a video using the Docker command."""

    index, video_path = video_info
    average_motion_vector_strength = 0.0
    motion_score = 0.0
    vmaf_score = 0.0

    if GET_VMAF_MOTION_SCORE:
        try:
            vmaf_cmd = VMAF_CMD.replace("$SRC_VIDEO$", video_path).replace(
                "$OUTPUT_PATH$", f"{TEMP_FOLDER}/output_{index}.json"
            )
            subprocess.run(vmaf_cmd, shell=True, check=True)
        except Exception as e:
            print(f"Error processing video {index}: {e}")
            return index, motion_score, vmaf_score, average_motion_vector_strength
        finally:
            # Read the output JSON file
            output_file = f"{TEMP_FOLDER}/output_{index}.json"
            if os.path.exists(output_file):
                with open(output_file) as f:
                    data = json.load(f)
                    # ref: https://video.stackexchange.com/questions/24210/how-should-i-interpret-the-results-from-netflix-vmaf
                    motion_score = (
                        data["pooled_metrics"]["integer_motion"]["mean"]
                        + data["pooled_metrics"]["integer_motion2"]["mean"]
                    ) / 2.0
                    vmaf_score = data["pooled_metrics"]["vmaf"]["mean"]

                os.remove(output_file)

    if GET_MOTION_VECTOR_AVERAGE:
        try:
            command = f'{EXTRACT_MVS_RUNNABLE_PATH} "{video_path}" > {TEMP_FOLDER}/{index}_motion_vectors.csv'
            subprocess.run(command, shell=True, check=True)
            average_motion_vector_strength = get_average_motion_vector_strength(
                f"{TEMP_FOLDER}/{index}_motion_vectors.csv"
            )
        except Exception as e:
            print(f"Error processing video {index}: {e}")
        finally:
            if os.path.exists(f"{TEMP_FOLDER}/{index}_motion_vectors.csv"):
                os.remove(f"{TEMP_FOLDER}/{index}_motion_vectors.csv")

    return index, motion_score, vmaf_score, average_motion_vector_strength


def process_videos_parallel(csv_file, num_processes=100):
    """Processes videos in parallel."""
    # Read CSV file
    df = pd.read_csv(csv_file)

    if "motion_score" not in df.columns and GET_VMAF_MOTION_SCORE:
        df["motion_score"] = 0.0
        df["vmaf_score"] = 0.0
    if "average_motion_vector_strength" not in df.columns and GET_MOTION_VECTOR_AVERAGE:
        df["average_motion_vector_strength"] = 0.0

    # Create temp folder if it doesn't exist
    create_temp_folder()

    # Create a pool of workers to process videos in parallel

    with multiprocessing.Pool(num_processes) as pool:
        # Each worker processes one video
        for result in list(tqdm.tqdm(pool.imap(process_video, enumerate(df["path"])), total=len(df))):
            index, motion_score, vmaf_score, average_motion_vector_strength = result
            # Update the dataframe with the motion score
            if GET_VMAF_MOTION_SCORE:
                df.at[index, "motion_score"] = motion_score
                df.at[index, "vmaf_score"] = vmaf_score
            if GET_MOTION_VECTOR_AVERAGE:
                df.at[index, "average_motion_vector_strength"] = average_motion_vector_strength

    # Remove temp folder after processing
    if os.path.exists(TEMP_FOLDER):
        shutil.rmtree(TEMP_FOLDER)

    # Save the updated dataframe to a new CSV file
    df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    args = parser.parse_args()
    process_videos_parallel(args.csv_file)
