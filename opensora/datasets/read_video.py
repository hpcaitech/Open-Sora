import gc
import math
import os
from fractions import Fraction
from typing import Any, Dict, Optional, Tuple, Union

import av
import cv2
import numpy as np
import torch
from torchvision.io.video import (
    _align_audio_frames,
    _check_av_available,
    _log_api_usage_once,
    _read_from_stream,
    _video_opt,
)


def read_video_av(
    filename: str,
    start_pts: Union[float, Fraction] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "pts",
    output_format: str = "THWC",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Reads a video from a file, returning both the video frames and the audio frames

    Args:
        filename (str): path to the video file
        start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The start presentation time of the video
        end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The end presentation time
        pts_unit (str, optional): unit in which start_pts and end_pts values will be interpreted,
            either 'pts' or 'sec'. Defaults to 'pts'.
        output_format (str, optional): The format of the output video tensors. Can be either "THWC" (default) or "TCHW".

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels and `L` is the number of points
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(read_video)

    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")

    from torchvision import get_video_backend

    if not os.path.exists(filename):
        raise RuntimeError(f"File not found: {filename}")

    if get_video_backend() != "pyav":
        vframes, aframes, info = _video_opt._read_video(filename, start_pts, end_pts, pts_unit)
    else:
        _check_av_available()

        if end_pts is None:
            end_pts = float("inf")

        if end_pts < start_pts:
            raise ValueError(
                f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}"
            )

        info = {}
        video_frames = []
        audio_frames = []
        audio_timebase = _video_opt.default_timebase

        container = av.open(filename, metadata_errors="ignore")
        try:
            if container.streams.audio:
                audio_timebase = container.streams.audio[0].time_base
            if container.streams.video:
                video_frames = _read_from_stream(
                    container,
                    start_pts,
                    end_pts,
                    pts_unit,
                    container.streams.video[0],
                    {"video": 0},
                )
                video_fps = container.streams.video[0].average_rate
                # guard against potentially corrupted files
                if video_fps is not None:
                    info["video_fps"] = float(video_fps)

            if container.streams.audio:
                audio_frames = _read_from_stream(
                    container,
                    start_pts,
                    end_pts,
                    pts_unit,
                    container.streams.audio[0],
                    {"audio": 0},
                )
                info["audio_fps"] = container.streams.audio[0].rate
        except av.AVError:
            # TODO raise a warning?
            pass
        finally:
            container.close()
            del container
            # NOTE: manually garbage collect to close pyav threads
            gc.collect()

        vframes_list = [frame.to_rgb().to_ndarray() for frame in video_frames]
        aframes_list = [frame.to_ndarray() for frame in audio_frames]

        if vframes_list:
            vframes = torch.as_tensor(np.stack(vframes_list))
        else:
            vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

        if aframes_list:
            aframes = np.concatenate(aframes_list, 1)
            aframes = torch.as_tensor(aframes)
            if pts_unit == "sec":
                start_pts = int(math.floor(start_pts * (1 / audio_timebase)))
                if end_pts != float("inf"):
                    end_pts = int(math.ceil(end_pts * (1 / audio_timebase)))
            aframes = _align_audio_frames(aframes, audio_frames, start_pts, end_pts)
        else:
            aframes = torch.empty((1, 0), dtype=torch.float32)

    if output_format == "TCHW":
        # [T,H,W,C] --> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    return vframes, aframes, info


def read_video_cv2(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        # print("Error: Unable to open video")
        raise ValueError
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        vinfo = {
            "video_fps": fps,
        }

        frames = []
        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # If frame is not read correctly, break the loop
            if not ret:
                break

            frames.append(frame[:, :, ::-1])  # BGR to RGB

            # Exit if 'q' is pressed
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()

        frames = np.stack(frames)
        frames = torch.from_numpy(frames)  # [T, H, W, C=3]
        frames = frames.permute(0, 3, 1, 2)
        return frames, vinfo


def read_video(video_path, backend="av"):
    if backend == "cv2":
        vframes, vinfo = read_video_cv2(video_path)
    elif backend == "av":
        vframes, _, vinfo = read_video_av(filename=video_path, pts_unit="sec", output_format="TCHW")
    else:
        raise ValueError

    return vframes, vinfo


if __name__ == "__main__":
    vframes, vinfo = read_video("./data/colors/9.mp4", backend="cv2")
    x = 0
