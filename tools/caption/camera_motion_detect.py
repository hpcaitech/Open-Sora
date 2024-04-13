# ref: https://github.com/antiboredom/camera-motion-detector

import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


def apply(df, func, **kwargs):
    if pandas_has_parallel:
        return df.parallel_apply(func, **kwargs)
    return df.progress_apply(func, **kwargs)


try:
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=True)
    pandas_has_parallel = True
except ImportError:
    pandas_has_parallel = False


def make_empty(new_w, new_h):
    empty = []
    for y in range(new_h):
        xvals = []
        for x in range(new_w):
            xvals.append([x, y])
        empty.append(xvals)

    empty = np.array(empty)
    return empty


def get_type(mag, ang, zoom_in, tau_static=1.0, tau_zoom=(0.4, 0.6)):
    if mag < tau_static:
        return "static"
    if zoom_in < tau_zoom[0]:
        return "zoom out"
    if zoom_in > tau_zoom[1]:
        return "zoom in"
    if ang < 45 or ang >= 315:
        return "pan left"
    if 45 <= ang < 135:
        return "tilt up"
    if 135 <= ang < 225:
        return "pan right"
    if 225 <= ang < 315:
        return "tilt down"
    return "unknown"


def get_video_type(frame_types):
    # count the number of each type
    counts = {}
    max_count = 0
    max_type = None
    for frame_type in frame_types:
        if frame_type not in counts:
            counts[frame_type] = 0
        counts[frame_type] += 1
        if counts[frame_type] > max_count:
            max_count = counts[frame_type]
            max_type = frame_type
    if max_count > len(frame_types) / 2:
        return max_type
    if "static" in counts:
        return "unknown"
    if "zoom in" not in counts and "zoom out" not in counts:
        return "pan/tilt"
    return "dynamic"


def process(path: str, frame_interval=15) -> str:
    cap = cv2.VideoCapture(path)
    count = 0
    prvs = None
    frame_types = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if count == 0:
                prvs = frame
                h, w = frame.shape
                empty = make_empty(w, h)
                empty_dists = np.sqrt(
                    np.square(empty.ravel()[::2] - (w / 2)) + np.square(empty.ravel()[1::2] - (h / 2))
                )
            else:
                flow = cv2.calcOpticalFlowFarneback(prvs, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
                mean_mag = np.median(mag)
                mean_ang = np.median(ang)

                flow_coords = flow + empty
                xvals = flow_coords.ravel()[::2] - (w / 2)
                yvals = flow_coords.ravel()[1::2] - (h / 2)
                dists = np.sqrt(np.square(xvals) + np.square(yvals))
                dist_diff = dists >= empty_dists
                zoom_in_factor = np.count_nonzero(dist_diff) / len(dist_diff)
                frame_types.append(get_type(mean_mag, mean_ang, zoom_in_factor))
            count += frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        else:
            cap.release()
            break
    video_type = get_video_type(frame_types)
    return video_type


def main(args):
    output_file = args.input.replace(".csv", "_cmotion.csv")
    data = pd.read_csv(args.input)
    data["cmotion"] = apply(data["path"], process)
    data.to_csv(output_file, index=False)
    print(f"Output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--disable-parallel", action="store_true")
    args = parser.parse_args()
    if args.disable_parallel:
        pandas_has_parallel = False
    main(args)
