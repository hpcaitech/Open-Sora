import os

import cv2
import numpy as np
from PIL import Image

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def is_video(filename):
    ext = os.path.splitext(filename)[-1].lower()
    return ext in VID_EXTENSIONS


def extract_frames(
    video_path,
    frame_inds=None,
    points=None,
    backend="opencv",
    return_length=False,
    num_frames=None,
):
    """
    Args:
        video_path (str): path to video
        frame_inds (List[int]): indices of frames to extract
        points (List[float]): values within [0, 1); multiply #frames to get frame indices
    Return:
        List[PIL.Image]
    """
    assert backend in ["av", "opencv", "decord"]
    assert (frame_inds is None) or (points is None)

    if backend == "av":
        import av

        container = av.open(video_path)
        if num_frames is not None:
            total_frames = num_frames
        else:
            total_frames = container.streams.video[0].frames

        if points is not None:
            frame_inds = [int(p * total_frames) for p in points]

        frames = []
        for idx in frame_inds:
            if idx >= total_frames:
                idx = total_frames - 1
            target_timestamp = int(idx * av.time_base / container.streams.video[0].average_rate)
            container.seek(target_timestamp)
            frame = next(container.decode(video=0)).to_image()
            frames.append(frame)

        if return_length:
            return frames, total_frames
        return frames

    elif backend == "decord":
        import decord

        container = decord.VideoReader(video_path, num_threads=1)
        if num_frames is not None:
            total_frames = num_frames
        else:
            total_frames = len(container)

        if points is not None:
            frame_inds = [int(p * total_frames) for p in points]

        frame_inds = np.array(frame_inds).astype(np.int32)
        frame_inds[frame_inds >= total_frames] = total_frames - 1
        frames = container.get_batch(frame_inds).asnumpy()  # [N, H, W, C]
        frames = [Image.fromarray(x) for x in frames]

        if return_length:
            return frames, total_frames
        return frames

    elif backend == "opencv":
        cap = cv2.VideoCapture(video_path)
        if num_frames is not None:
            total_frames = num_frames
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if points is not None:
            frame_inds = [int(p * total_frames) for p in points]

        frames = []
        for idx in frame_inds:
            if idx >= total_frames:
                idx = total_frames - 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

            # HACK: sometimes OpenCV fails to read frames, return a black frame instead
            try:
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
            except Exception as e:
                print(f"[Warning] Error reading frame {idx} from {video_path}: {e}")
                # First, try to read the first frame
                try:
                    print(f"[Warning] Try reading first frame.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                # If that fails, return a black frame
                except Exception as e:
                    print(f"[Warning] Error in reading first frame from {video_path}: {e}")
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame = Image.new("RGB", (width, height), (0, 0, 0))

            # HACK: if height or width is 0, return a black frame instead
            if frame.height == 0 or frame.width == 0:
                height = width = 256
                frame = Image.new("RGB", (width, height), (0, 0, 0))

            frames.append(frame)

        if return_length:
            return frames, total_frames
        return frames
    else:
        raise ValueError
