import numpy as np
import torch
from decord import VideoReader
from PIL import Image, ImageSequence


def get_frame_indices(num_frames, vlen, sample="rand", fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]:  # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == "middle":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[: len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


def load_video(video_path, data_transform=None, num_frames=None, return_tensor=True, width=None, height=None):
    """
    Load a video from a given path and apply optional data transformations.

    The function supports loading video in GIF (.gif), PNG (.png), and MP4 (.mp4) formats.
    Depending on the format, it processes and extracts frames accordingly.

    Parameters:
    - video_path (str): The file path to the video or image to be loaded.
    - data_transform (callable, optional): A function that applies transformations to the video data.

    Returns:
    - frames (torch.Tensor): A tensor containing the video frames with shape (T, C, H, W),
      where T is the number of frames, C is the number of channels, H is the height, and W is the width.

    Raises:
    - NotImplementedError: If the video format is not supported.

    The function first determines the format of the video file by its extension.
    For GIFs, it iterates over each frame and converts them to RGB.
    For PNGs, it reads the single frame, converts it to RGB.
    For MP4s, it reads the frames using the VideoReader class and converts them to NumPy arrays.
    If a data_transform is provided, it is applied to the buffer before converting it to a tensor.
    Finally, the tensor is permuted to match the expected (T, C, H, W) format.
    """
    if video_path.endswith(".gif"):
        frame_ls = []
        img = Image.open(video_path)
        for frame in ImageSequence.Iterator(img):
            frame = frame.convert("RGB")
            frame = np.array(frame).astype(np.uint8)
            frame_ls.append(frame)
        buffer = np.array(frame_ls).astype(np.uint8)
    elif video_path.endswith(".png"):
        frame = Image.open(video_path)
        frame = frame.convert("RGB")
        frame = np.array(frame).astype(np.uint8)
        frame_ls = [frame]
        buffer = np.array(frame_ls)
    elif video_path.endswith(".mp4"):
        import decord

        decord.bridge.set_bridge("native")
        if width:
            video_reader = VideoReader(video_path, width=width, height=height, num_threads=1)
        else:
            video_reader = VideoReader(video_path, num_threads=1)
        frames = video_reader.get_batch(range(len(video_reader)))  # (T, H, W, C), torch.uint8

        buffer = frames.asnumpy().astype(np.uint8)
    else:
        raise NotImplementedError

    frames = buffer
    if num_frames:
        frame_indices = get_frame_indices(num_frames, len(frames), sample="middle")
        frames = frames[frame_indices]

    if data_transform:
        frames = data_transform(frames)
    elif return_tensor:
        frames = torch.Tensor(frames)
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8

    return frames
