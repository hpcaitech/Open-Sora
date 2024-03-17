import os
import subprocess

import cv2
from imageio_ffmpeg import get_ffmpeg_exe
from mmengine.logging import print_log
from moviepy.editor import VideoFileClip
from scenedetect import FrameTimecode


def iterate_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        # root contains the current directory path
        # dirs contains the list of subdirectories in the current directory
        # files contains the list of files in the current directory

        # Process files in the current directory
        for file in files:
            file_path = os.path.join(root, file)
            # print("File:", file_path)
            yield file_path

        # Process subdirectories and recursively call the function
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            # print("Subdirectory:", subdir_path)
            iterate_files(subdir_path)


def iterate_folders(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            yield subdir_path
            # print("Subdirectory:", subdir_path)
            iterate_folders(subdir_path)


def clone_folder_structure(root_src, root_dst, verbose=False):
    src_path_list = iterate_folders(root_src)
    src_relpath_list = [os.path.relpath(x, root_src) for x in src_path_list]

    os.makedirs(root_dst, exist_ok=True)
    dst_path_list = [os.path.join(root_dst, x) for x in src_relpath_list]
    for folder_path in dst_path_list:
        os.makedirs(folder_path, exist_ok=True)
        if verbose:
            print(f"Create folder: '{folder_path}'")


def count_files(root, suffix=".mp4"):
    files_list = iterate_files(root)
    cnt = len([x for x in files_list if x.endswith(suffix)])
    return cnt


def check_mp4_integrity(file_path, verbose=True, logger=None):
    try:
        VideoFileClip(file_path)
        if verbose:
            print_log(f"The MP4 file '{file_path}' is intact.", logger=logger)
        return True
    except Exception as e:
        if verbose:
            print_log(f"Error: {e}", logger=logger)
            print_log(f"The MP4 file '{file_path}' is not intact.", logger=logger)
        return False


def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in the video '{video_path}': {total_frames}")

    cap.release()


def split_video(
    sample_path,
    scene_list,
    save_dir,
    target_fps=30,
    min_seconds=1,
    max_seconds=10,
    shorter_size=512,
    verbose=False,
    logger=None,
):
    FFMPEG_PATH = get_ffmpeg_exe()

    save_path_list = []
    for idx, scene in enumerate(scene_list):
        s, t = scene  # FrameTimecode
        fps = s.framerate
        max_duration = FrameTimecode(timecode="00:00:00", fps=fps)
        max_duration.frame_num = round(fps * max_seconds)
        duration = min(max_duration, t - s)
        if duration.get_frames() < round(min_seconds * fps):
            continue

        # save path
        fname = os.path.basename(sample_path)
        fname_wo_ext = os.path.splitext(fname)[0]
        # TODO: fname pattern
        save_path = os.path.join(save_dir, f"{fname_wo_ext}_scene-{idx}.mp4")

        # ffmpeg cmd
        cmd = [FFMPEG_PATH]

        # Only show ffmpeg output for the first call, which will display any
        # errors if it fails, and then break the loop. We only show error messages
        # for the remaining calls.
        # cmd += ['-v', 'error']

        # input path
        cmd += ["-i", sample_path]

        # clip to cut
        cmd += ["-nostdin", "-y", "-ss", str(s.get_seconds()), "-t", str(duration.get_seconds())]

        # target fps
        # cmd += ['-vf', 'select=mod(n\,2)']
        cmd += ["-r", f"{target_fps}"]

        # aspect ratio
        cmd += ["-vf", f"scale='if(gt(iw,ih),-2,{shorter_size})':'if(gt(iw,ih),{shorter_size},-2)'"]
        # cmd += ['-vf', f"scale='if(gt(iw,ih),{shorter_size},trunc(ow/a/2)*2)':-2"]

        cmd += ["-map", "0", save_path]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = proc.communicate()
        if verbose:
            stdout = stdout.decode("utf-8")
            print_log(stdout, logger=logger)

        save_path_list.append(sample_path)
        print_log(f"Video clip saved to '{save_path}'", logger=logger)

    return save_path_list
