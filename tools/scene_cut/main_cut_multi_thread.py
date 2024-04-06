import os
import argparse
import time
import subprocess
from tqdm import tqdm

import pandas as pd
from scenedetect import FrameTimecode
from imageio_ffmpeg import get_ffmpeg_exe
from concurrent.futures import ThreadPoolExecutor, as_completed

from mmengine.logging import MMLogger, print_log
from utils_video import is_intact_video, iterate_files, clone_folder_structure


def single_process(row, save_dir, logger=None):
    # video_id = row['videoID']
    # video_path = os.path.join(root_src, f'{video_id}.mp4')
    video_path = row['path']

    # check mp4 integrity
    # if not is_intact_video(video_path, logger=logger):
    #     return False

    timestamp = row['timestamp']
    if not (timestamp.startswith('[') and timestamp.endswith(']')):
        return False
    scene_list = eval(timestamp)
    scene_list = [
        (FrameTimecode(s, fps=1), FrameTimecode(t, fps=1))
        for s, t in scene_list
    ]
    split_video(video_path, scene_list, save_dir=save_dir, logger=logger)
    return True


def split_video(
        video_path,
        scene_list,
        save_dir,
        min_seconds=None,
        max_seconds=None,
        target_fps=30,
        shorter_size=512,
        verbose=False,
        logger=None,
):
    """
    scenes shorter than min_seconds will be ignored;
    scenes longer than max_seconds will be cut to save the beginning max_seconds.
    Currently, the saved file name pattern is f'{fname}_scene-{idx}'.mp4

    Args:
        scene_list (List[Tuple[FrameTimecode, FrameTimecode]]): each element is (s, t): start and end of a scene.
        min_seconds (float | None)
        max_seconds (float | None)
        target_fps (int | None)
        shorter_size (int | None)
    """
    FFMPEG_PATH = get_ffmpeg_exe()

    save_path_list = []
    for idx, scene in enumerate(scene_list):
        s, t = scene  # FrameTimecode
        if min_seconds is not None:
            if (t - s).get_seconds() < min_seconds:
                continue

        duration = t - s
        if max_seconds is not None:
            fps = s.framerate
            max_duration = FrameTimecode(timecode="00:00:00", fps=fps)
            max_duration.frame_num = round(fps * max_seconds)
            duration = min(max_duration, duration)

        # save path
        fname = os.path.basename(video_path)
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
        # cmd += ["-i", video_path]

        # clip to cut
        cmd += ["-nostdin", "-y", "-ss", str(s.get_seconds()), "-i", video_path, "-t", str(duration.get_seconds())]
        # cmd += ["-nostdin", "-y", "-ss", str(s.get_seconds()), "-t", str(duration.get_seconds())]

        # target fps
        # cmd += ['-vf', 'select=mod(n\,2)']
        if target_fps is not None:
            cmd += ["-r", f"{target_fps}"]

        # aspect ratio
        if shorter_size is not None:
            cmd += ["-vf", f"scale='if(gt(iw,ih),-2,{shorter_size})':'if(gt(iw,ih),{shorter_size},-2)'"]
        # cmd += ['-vf', f"scale='if(gt(iw,ih),{shorter_size},trunc(ow/a/2)*2)':-2"]

        cmd += ["-map", "0", save_path]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = proc.communicate()
        if verbose:
            stdout = stdout.decode("utf-8")
            print_log(stdout, logger=logger)

        save_path_list.append(video_path)
        print_log(f"Video clip saved to '{save_path}'", logger=logger)

    return save_path_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='F:/Panda-70M/')
    parser.add_argument('--split', default='test')
    parser.add_argument('--num_workers', default=5, type=int)

    args = parser.parse_args()
    return args


def main():
    # args = parse_args()
    # root = args.root
    # split = args.split

    root = 'F:/Panda-70M/'
    root, split = 'F:/pexels_new/', 'popular_2'
    meta_path = os.path.join(root, f'raw/meta/{split}_format_timestamp.csv')
    root_dst = os.path.join(root, f'scene_cut/data/{split}')

    folder_dst = root_dst
    # folder_src = os.path.join(root_src, f'data/{split}')
    # folder_dst = os.path.join(root_dst, os.path.relpath(folder_src, root_src))
    os.makedirs(folder_dst, exist_ok=True)

    meta = pd.read_csv(meta_path)

    # create logger
    # folder_path_log = os.path.dirname(root_dst)
    # log_name = os.path.basename(root_dst)
    # timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))
    # log_path = os.path.join(folder_path_log, f"{log_name}_{timestamp}.log")
    # logger = MMLogger.get_instance(log_name, log_file=log_path)
    logger = None

    tasks = []
    pool = ThreadPoolExecutor(max_workers=1)
    for idx, row in meta.iterrows():
        task = pool.submit(single_process, row, folder_dst, logger)
        tasks.append(task)

    for task in tqdm(as_completed(tasks), total=len(meta)):
        task.result()
    pool.shutdown()


if __name__ == '__main__':
    main()
