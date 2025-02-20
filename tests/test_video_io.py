import os

import cv2
import numpy as np
import torch
from mmengine.runner import set_random_seed
from tqdm import tqdm

from opensora.datasets import video_transforms
from opensora.datasets.utils import read_file
from opensora.utils.misc import to_torch_dtype

# data_path = "~/data/issue.csv"
data_path = "~/data/test.csv"
save_dir = "samples/debug_original_video_read_write"
num_frames = 17
frame_interval = 1
image_size = 1024

set_random_seed(1024)
os.makedirs(save_dir, exist_ok=True)
data = read_file(data_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = to_torch_dtype("bf16")


def temporal_random_crop(vframes, num_frames, frame_interval):
    temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
    total_frames = len(vframes)
    start_frame_ind, end_frame_ind = temporal_sample(total_frames)
    assert (
        end_frame_ind - start_frame_ind >= num_frames
    ), f"Not enough frames to sample, {end_frame_ind} - {start_frame_ind} < {num_frames}"
    frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, num_frames, dtype=int)
    video = vframes[frame_indice]
    return video


def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    """
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    # breakpoint()
    return clip.float() / 255.0


def read_video_cv2(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
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

            # frames.append(frame[:, :, ::-1])  # BGR to RGB
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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


def write_video_cv2(path, video, fps=24, image_size=(1920, 1080)):
    # Set the video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(path, fourcc, fps, image_size)

    for frame_idx in range(video.size(0)):
        frame = np.array(video[frame_idx].permute(1, 2, 0))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output.write(frame)

    output.release()


for video_path in tqdm(data["path"]):
    name = os.path.basename(video_path)

    # # DEBUG: read image and save as if video: no issue
    # image = cv2.imread('/home/shenchenhui/data/ship-in-coffee-image.png')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = torch.Tensor(image)
    # fake_vid = image.repeat(48, 1, 1, 1)
    # write_video(f"{save_dir}/fake.mp4", fake_vid, fps=24, video_codec="h264")

    # # ===== data loading ====== #
    # # loading
    # vframes, vinfo = read_video(video_path, backend="cv2")
    vframes, vinfo = read_video_cv2(video_path)
    video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24
    print("fps:", video_fps)
    # # Sampling video frames
    video = vframes
    video = temporal_random_crop(vframes, num_frames, frame_interval)  # not this issue
    # # breakpoint()

    # video = to_tensor(video)

    # # # transform
    # # transform_video = transforms.Compose(
    # #     [
    # #         # video_transforms.ToTensorVideo(),  # moved up
    # #         # video_transforms.UCFCenterCropVideo(image_size), # not this issue
    # #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True), # not this issues
    # #     ]
    # # )
    # # video = transform_video(video)  # T C H W

    ### write each frame as image
    # for frame_idx in range(video.size(0)):
    #     frame = np.array(video[frame_idx].permute(1,2,0))
    #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(f'{save_dir}/{frame_idx}.jpg', frame)

    # # # TCHW -> CTHW
    # video = video.permute(1, 0, 2, 3)

    # # # # ===== model training ====== #
    # # # video = video.to(device, dtype)

    # # # ===== data saving ====== #
    # # # # Normalize
    # # # low, high = -1,1
    # # # video.clamp_(min=low, max=high)
    # # # video.sub_(low).div_(max(high - low, 1e-5))

    # # # video = video.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8)
    # # # breakpoint()
    # # # video = video.permute(1, 2, 3, 0).to("cpu", torch.uint8)
    # video = video.permute(1, 2, 3, 0)

    #
    write_video_cv2(f"{save_dir}/{name}", video, fps=video_fps)
    # # prep to [T, H, W, C] in order to write
    # write_video(f"{save_dir}/{name}", video, fps=video_fps, video_codec="h264")
