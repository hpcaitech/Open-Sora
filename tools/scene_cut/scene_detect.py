import argparse
import os
from functools import partial

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from scenedetect import AdaptiveDetector, SceneManager, StatsManager, open_video
from tqdm import tqdm

tqdm.pandas()


def detect(
    video_path,
    detector,
    backend="pyav",
    stats_file_path=None,
    show_progress=False,
    start_time=None,
    end_time=None,
    start_in_scene=False,
):
    """
    Adapted from scenedetect.detect()
    Modifications:
        - allow passing backend to open_video()
    """
    video = open_video(video_path, backend=backend)
    if start_time is not None:
        start_time = video.base_timecode + start_time
        video.seek(start_time)
    if end_time is not None:
        end_time = video.base_timecode + end_time
    # To reduce memory consumption when not required, we only add a StatsManager if we
    # need to save frame metrics to disk.
    scene_manager = SceneManager(StatsManager() if stats_file_path else None)
    scene_manager.add_detector(detector)
    scene_manager.detect_scenes(
        video=video,
        show_progress=show_progress,
        end_time=end_time,
    )
    if not scene_manager.stats_manager is None:
        scene_manager.stats_manager.save_to_csv(csv_file=stats_file_path)
    return scene_manager.get_scene_list(start_in_scene=start_in_scene)


def detect_transition(
    video_path,
    backend="pyav",
    transition_seconds=1.0,
    start_time=None,
    end_time=None,
    start_in_scene=False,
    stats_file_path=None,
    show_progress=False,
):
    video = open_video(video_path, backend=backend)
    fps = video.frame_rate

    frame_skip = int(transition_seconds * fps)
    t_ada, t_con = 2.0, 10.0
    window = 1
    if end_time is not None:
        end_time = video.base_timecode + end_time

    # 1. detect from the very beginning
    detector = AdaptiveDetector(
        adaptive_threshold=t_ada,  # default 3.0
        min_content_val=t_con,  # default 15.0
        min_scene_len=1,
        window_width=window,
    )
    scene_manager = SceneManager(StatsManager() if stats_file_path else None)
    scene_manager.add_detector(detector)

    if start_time is not None:
        tmp = video.base_timecode + start_time
        video.seek(tmp)

    scene_manager.detect_scenes(
        video=video,
        show_progress=show_progress,
        frame_skip=frame_skip,
        end_time=end_time,
    )
    if scene_manager.stats_manager is not None:
        scene_manager.stats_manager.save_to_csv(csv_file=stats_file_path)
    s0 = scene_manager.get_scene_list(start_in_scene=start_in_scene)

    # 2. detect from half transition_seconds
    detector = AdaptiveDetector(
        adaptive_threshold=t_ada,  # default 3.0
        min_content_val=t_con,  # default 15.0
        min_scene_len=1,
        window_width=window,
    )
    scene_manager = SceneManager(StatsManager() if stats_file_path else None)
    scene_manager.add_detector(detector)

    tmp = video.base_timecode + transition_seconds / 2
    if start_time is not None:
        tmp = tmp + start_time
    video.seek(tmp)

    scene_manager.detect_scenes(
        video=video,
        show_progress=show_progress,
        frame_skip=frame_skip,
        end_time=end_time,
    )
    if scene_manager.stats_manager is not None:
        scene_manager.stats_manager.save_to_csv(csv_file=f"{stats_file_path}.csv")
    s1 = scene_manager.get_scene_list(start_in_scene=start_in_scene)

    # merge s0, s1
    s0_s = [x[0] for x in s0]
    s1_t = [x[1] for x in s1]
    i, j = 0, 0
    merged = []
    while i < len(s0_s) and j < len(s1_t):
        l, r = s0_s[i], s1_t[j]
        if l < r:
            merged.append(l)
            i += 1
        elif l == r:
            merged.append(l)
            i += 1
            j += 1
        else:
            merged.append(r)
            j += 1
    merged.extend(s0_s[i:])
    merged.extend(s1_t[j:])

    # remove transitions
    scene_list = []
    m = transition_seconds
    for idx in range(len(merged) - 1):
        cur = merged[idx]
        next = merged[idx + 1]
        if next - cur < m:
            continue

        if idx + 2 < len(merged) and (merged[idx + 2] - next) < m:
            # detected by both s0 & s1
            next = next - m
        elif idx + 2 < len(merged):
            next = next - 1.5 * m
        if idx - 1 >= 0 and (cur - merged[idx - 1]) < m:
            # detected by both s0 & s1
            cur = cur + 0.5 * m
        elif idx > 0:
            cur = cur + 0.5 * m

        if cur < next:
            scene_list.append((cur, next))

    return scene_list


def process_single_row(row, args):
    video_path = row["path"]

    try:
        if args.transition_seconds is not None:
            scene_list = detect_transition(
                video_path,
                transition_seconds=args.transition_seconds,
                start_time=args.start_time,
                end_time=args.end_time,
                start_in_scene=True,
            )
            timestamp_intact = [(s.get_timecode(), t.get_timecode()) for s, t in scene_list]
        else:
            detector = AdaptiveDetector(
                adaptive_threshold=2.0,
                min_content_val=10.0,
                min_scene_len=1,
            )
            scene_list = detect(
                video_path,
                detector,
                start_time=args.start_time,
                end_time=args.end_time,
                start_in_scene=True,
            )
            margin = 1.0 / scene_list[0][0].framerate
            scene_list = [(s, t - margin) for s, t in scene_list if s < t - margin]
            timestamp_intact = [(s.get_timecode(), t.get_timecode()) for s, t in scene_list]

        if args.max_seconds is not None:
            timestamp_cut = []
            for s, t in scene_list:
                while (t - s).get_seconds() > args.max_seconds:
                    tmp = s + args.max_seconds
                    timestamp_cut.append((s.get_timecode(), tmp.get_timecode()))
                    s = tmp
                timestamp_cut.append((s.get_timecode(), t.get_timecode()))
        else:
            timestamp_cut = timestamp_intact
        return True, str(timestamp_cut), str(timestamp_intact)

    except Exception as e:
        print(f"Video '{video_path}' with error {e}")
        return False, "", ""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str)
    parser.add_argument(
        "--max_seconds", type=float, default=None, help="if not None, any single timestamp will not exceed max_seconds"
    )
    parser.add_argument("--transition_seconds", type=float, default=None, help="if not None, use detect_transition()")
    parser.add_argument("--start_time", type=float, default=None, help="if not None, start detection from start_time")
    parser.add_argument("--end_time", type=float, default=None, help="if not None, end detection at end_time")
    parser.add_argument("--num_workers", type=int, default=None, help="#workers for pandarallel")
    parser.add_argument("--disable_parallel", action="store_true", help="disable parallel processing")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.transition_seconds is not None:
        assert args.transition_seconds > 0

    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        exit()

    # initialize pandarallel
    if not args.disable_parallel:
        if args.num_workers is not None:
            pandarallel.initialize(progress_bar=True, nb_workers=args.num_workers)
        else:
            pandarallel.initialize(progress_bar=True)
    process_single_row_partial = partial(process_single_row, args=args)

    meta = pd.read_csv(meta_path)
    if not args.disable_parallel:
        ret = meta.parallel_apply(process_single_row_partial, axis=1)
    else:
        ret = meta.apply(process_single_row_partial, axis=1)

    succ, timestamp_cut, timestamp_intact = list(zip(*ret))
    meta["timestamp"] = timestamp_cut
    meta["timestamp_intact"] = timestamp_intact
    meta = meta[np.array(succ)]

    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_timestamp{ext}"
    meta.to_csv(out_path, index=False)
    print(f"New meta (shape={meta.shape}) with timestamp saved to '{out_path}'.")


if __name__ == "__main__":
    main()
