import argparse
import os
import subprocess
import time
from functools import partial

import pandas as pd
from imageio_ffmpeg import get_ffmpeg_exe
from mmengine.logging import MMLogger, print_log
from pandarallel import pandarallel
from scenedetect import FrameTimecode
from tqdm import tqdm

tqdm.pandas()


def process_single_row(row, args, log_name=None):
    video_path = row["path"]

    logger = None
    if log_name is not None:
        logger = MMLogger.get_instance(log_name)

    # check mp4 integrity
    # if not is_intact_video(video_path, logger=logger):
    #     return False

    timestamp = row["timestamp"]
    if not (timestamp.startswith("[") and timestamp.endswith("]")):
        return False
    scene_list = eval(timestamp)
    scene_list = [(FrameTimecode(s, fps=1), FrameTimecode(t, fps=1)) for s, t in scene_list]
    split_video(
        video_path,
        scene_list,
        save_dir=args.save_dir,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        target_fps=args.target_fps,
        shorter_size=args.shorter_size,
        logger=logger,
    )


def split_video(
    video_path,
    scene_list,
    save_dir,
    min_seconds=2.0,
    max_seconds=15.0,
    target_fps=30,
    shorter_size=720,
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

        # clip to cut
        # -ss after -i is very slow; put -ss before -i
        cmd += ["-nostdin", "-y", "-ss", str(s.get_seconds()), "-i", video_path, "-t", str(duration.get_seconds())]

        # target fps
        if target_fps is not None:
            cmd += ["-r", f"{target_fps}"]

        # aspect ratio
        if shorter_size is not None:
            cmd += ["-vf", f"scale='if(gt(iw,ih),-2,{shorter_size})':'if(gt(iw,ih),{shorter_size},-2)'"]
            # cmd += ['-vf', f"scale='if(gt(iw,ih),{shorter_size},trunc(ow/a/2)*2)':-2"]

        cmd += ["-map", "0", save_path]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = proc.communicate()
        # stdout = stdout.decode("utf-8")
        # print_log(stdout, logger=logger)

        save_path_list.append(video_path)
        if verbose:
            print_log(f"Video clip saved to '{save_path}'", logger=logger)

    return save_path_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--min_seconds", type=float, default=None,
                        help='if not None, clip shorter than min_seconds is ignored')
    parser.add_argument("--max_seconds", type=float, default=None,
                        help='if not None, clip longer than max_seconds is truncated')
    parser.add_argument("--target_fps", type=int, default=30, help='target fps of clips')
    parser.add_argument("--shorter_size", type=int, default=720, help='resize the shorter size by keeping ratio')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # create logger
    log_dir = os.path.dirname(save_dir)
    log_name = os.path.basename(save_dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))
    log_path = os.path.join(log_dir, f"{log_name}_{timestamp}.log")
    logger = MMLogger.get_instance(log_name, log_file=log_path)
    # logger = None

    # initialize pandarallel
    pandarallel.initialize(progress_bar=True)
    process_single_row_partial = partial(process_single_row, args=args, log_name=log_name)

    # process
    meta = pd.read_csv(args.meta_path)
    meta.parallel_apply(process_single_row_partial, axis=1)


if __name__ == "__main__":
    main()
