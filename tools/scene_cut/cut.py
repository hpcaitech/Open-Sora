import cv2  # isort:skip

import argparse
import os
import subprocess
from functools import partial

import pandas as pd
from imageio_ffmpeg import get_ffmpeg_exe
from pandarallel import pandarallel
from scenedetect import FrameTimecode
from tqdm import tqdm

tqdm.pandas()


def print_log(s, logger=None):
    if logger is not None:
        logger.info(s)
    else:
        print(s)


def process_single_row(row, args):
    video_path = row["path"]

    logger = None

    # check mp4 integrity
    # if not is_intact_video(video_path, logger=logger):
    #     return False

    if "timestamp" in row:
        timestamp = row["timestamp"]
        if not (timestamp.startswith("[") and timestamp.endswith("]")):
            return False
        scene_list = eval(timestamp)
        scene_list = [(FrameTimecode(s, fps=1), FrameTimecode(t, fps=1)) for s, t in scene_list]
    else:
        scene_list = [None]

    if "relpath" in row:
        save_dir = os.path.dirname(os.path.join(args.save_dir, row["relpath"]))
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = args.save_dir

    shorter_size = args.shorter_size
    if (shorter_size is not None) and ("height" in row) and ("width" in row):
        min_size = min(row["height"], row["width"])
        if min_size <= shorter_size:
            shorter_size = None

    split_video(
        video_path,
        scene_list,
        save_dir=save_dir,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        target_fps=args.target_fps,
        shorter_size=shorter_size,
        logger=logger,
    )


def split_video(
    video_path,
    scene_list,
    save_dir,
    min_seconds=2,
    max_seconds=15,
    target_fps=30,
    shorter_size=None,
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
        if scene is not None:
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
        # Note: -ss after -i is very slow; put -ss before -i !!!
        if scene is None:
            cmd += ["-nostdin", "-y", "-i", video_path]
        else:
            cmd += ["-nostdin", "-y", "-ss", str(s.get_seconds()), "-i", video_path, "-t", str(duration.get_seconds())]

        # target fps
        if target_fps is not None:
            cmd += ["-r", f"{target_fps}"]

        # aspect ratio
        if shorter_size is not None:
            cmd += ["-vf", f"scale='if(gt(iw,ih),-2,{shorter_size})':'if(gt(iw,ih),{shorter_size},-2)'"]
            # cmd += ['-vf', f"scale='if(gt(iw,ih),{shorter_size},trunc(ow/a/2)*2)':-2"]

        cmd += ["-map", "0:v", save_path]

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
    parser.add_argument(
        "--min_seconds", type=float, default=None, help="if not None, clip shorter than min_seconds is ignored"
    )
    parser.add_argument(
        "--max_seconds", type=float, default=None, help="if not None, clip longer than max_seconds is truncated"
    )
    parser.add_argument("--target_fps", type=int, default=None, help="target fps of clips")
    parser.add_argument(
        "--shorter_size", type=int, default=1080, help="resize the shorter size by keeping ratio; will not do upscale"
    )
    parser.add_argument("--num_workers", type=int, default=None, help="#workers for pandarallel")
    parser.add_argument("--disable_parallel", action="store_true", help="disable parallel processing")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        exit()

    # create logger
    os.makedirs(args.save_dir, exist_ok=True)

    # initialize pandarallel
    if not args.disable_parallel:
        if args.num_workers is not None:
            pandarallel.initialize(progress_bar=True, nb_workers=args.num_workers)
        else:
            pandarallel.initialize(progress_bar=True)
    process_single_row_partial = partial(process_single_row, args=args)

    # process
    meta = pd.read_csv(args.meta_path)
    if not args.disable_parallel:
        meta.parallel_apply(process_single_row_partial, axis=1)
    else:
        meta.apply(process_single_row_partial, axis=1)


if __name__ == "__main__":
    main()
