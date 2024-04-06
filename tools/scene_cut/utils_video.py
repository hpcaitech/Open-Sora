import os
import cv2
from mmengine.logging import print_log
from moviepy.editor import VideoFileClip


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


def is_intact_video(video_path, mode='moviepy', verbose=False, logger=None):
    if not os.path.exists(video_path):
        if verbose:
            print_log(f"Could not find '{video_path}'", logger=logger)
        return False

    if mode == 'moviepy':
        try:
            VideoFileClip(video_path)
            if verbose:
                print_log(f"The video file '{video_path}' is intact.", logger=logger)
            return True
        except Exception as e:
            if verbose:
                print_log(f"Error: {e}", logger=logger)
                print_log(f"The video file '{video_path}' is not intact.", logger=logger)
            return False
    elif mode == 'cv2':
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                if verbose:
                    print_log(f"The video file '{video_path}' is intact.", logger=logger)
                return True
        except Exception as e:
            if verbose:
                print_log(f"Error: {e}", logger=logger)
                print_log(f"The video file '{video_path}' is not intact.", logger=logger)
            return False
    else:
        raise ValueError


def count_frames(video_path, logger=None):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print_log(f"Error: Could not open video file '{video_path}'", logger=logger)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print_log(f"Total frames in the video '{video_path}': {total_frames}", logger=logger)

    cap.release()


def count_files(root, suffix=".mp4"):
    files_list = iterate_files(root)
    cnt = len([x for x in files_list if x.endswith(suffix)])
    return cnt

