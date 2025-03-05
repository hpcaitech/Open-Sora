import math
import os
import random
import re
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
from torchvision.io import write_video
from torchvision.utils import save_image

from . import video_transforms
from .read_video import read_video

try:
    import dask.dataframe as dd

    SUPPORT_DASK = True
except:
    SUPPORT_DASK = False

VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

regex = re.compile(
    r"^(?:http|ftp)s?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


def is_img(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext in IMG_EXTENSIONS


def is_vid(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext in VID_EXTENSIONS


def is_url(url):
    return re.match(regex, url) is not None


def read_file(input_path, memory_efficient=False):
    if input_path.endswith(".csv"):
        assert not memory_efficient, "Memory efficient mode is not supported for CSV files"
        return pd.read_csv(input_path)
    elif input_path.endswith(".parquet"):
        columns = None
        if memory_efficient:
            columns = ["path", "num_frames", "height", "width", "aspect_ratio", "fps", "resolution"]
        if SUPPORT_DASK:
            ret = dd.read_parquet(input_path, columns=columns).compute()
        else:
            ret = pd.read_parquet(input_path, columns=columns)
        return ret
    else:
        raise NotImplementedError(f"Unsupported file format: {input_path}")


def download_url(input_path):
    output_dir = "cache"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, base_name)
    img_data = requests.get(input_path).content
    with open(output_path, "wb", encoding="utf-8") as handler:
        handler.write(img_data)
    print(f"URL {input_path} downloaded to {output_path}")
    return output_path


def temporal_random_crop(
    vframes: torch.Tensor, num_frames: int, frame_interval: int, return_frame_indices: bool = False
) -> torch.Tensor | tuple[torch.Tensor, np.ndarray]:
    temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
    total_frames = len(vframes)
    start_frame_ind, end_frame_ind = temporal_sample(total_frames)

    assert (
        end_frame_ind - start_frame_ind >= num_frames
    ), f"Not enough frames to sample, {end_frame_ind} - {start_frame_ind} < {num_frames}"

    frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, num_frames, dtype=int)
    video = vframes[frame_indices]
    if return_frame_indices:
        return video, frame_indices
    else:
        return video


def get_transforms_video(name="center", image_size=(256, 256)):
    if name is None:
        return None
    elif name == "center":
        assert image_size[0] == image_size[1], "image_size must be square for center crop"
        transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                # video_transforms.RandomHorizontalFlipVideo(),
                video_transforms.UCFCenterCropVideo(image_size[0]),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    elif name == "resize_crop":
        transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                video_transforms.ResizeCrop(image_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    elif name == "rand_size_crop":
        transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                video_transforms.RandomSizedCrop(image_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    else:
        raise NotImplementedError(f"Transform {name} not implemented")
    return transform_video


def get_transforms_image(name="center", image_size=(256, 256)):
    if name is None:
        return None
    elif name == "center":
        assert image_size[0] == image_size[1], "Image size must be square for center crop"
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size[0])),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    elif name == "resize_crop":
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: resize_crop_to_fill(pil_image, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    elif name == "rand_size_crop":
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: rand_size_crop_arr(pil_image, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    else:
        raise NotImplementedError(f"Transform {name} not implemented")
    return transform


def read_image_from_path(path, transform=None, transform_name="center", num_frames=1, image_size=(256, 256)):
    image = pil_loader(path)
    if transform is None:
        transform = get_transforms_image(image_size=image_size, name=transform_name)
    image = transform(image)
    video = image.unsqueeze(0).repeat(num_frames, 1, 1, 1)
    video = video.permute(1, 0, 2, 3)
    return video


def read_video_from_path(path, transform=None, transform_name="center", image_size=(256, 256)):
    vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
    if transform is None:
        transform = get_transforms_video(image_size=image_size, name=transform_name)
    video = transform(vframes)  # T C H W
    video = video.permute(1, 0, 2, 3)
    return video


def read_from_path(path, image_size, transform_name="center"):
    if is_url(path):
        path = download_url(path)
    ext = os.path.splitext(path)[-1].lower()
    if ext.lower() in VID_EXTENSIONS:
        return read_video_from_path(path, image_size=image_size, transform_name=transform_name)
    else:
        assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
        return read_image_from_path(path, image_size=image_size, transform_name=transform_name)


def save_sample(
    x,
    save_path=None,
    fps=8,
    normalize=True,
    value_range=(-1, 1),
    force_video=False,
    verbose=True,
    crf=23,
):
    """
    Args:
        x (Tensor): shape [C, T, H, W]
    """
    assert x.ndim == 4

    if not force_video and x.shape[1] == 1:  # T = 1: save as image
        save_path += ".png"
        x = x.squeeze(1)
        save_image([x], save_path, normalize=normalize, value_range=value_range)
    else:
        save_path += ".mp4"
        if normalize:
            low, high = value_range
            x.clamp_(min=low, max=high)
            x.sub_(low).div_(max(high - low, 1e-5))

        x = x.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8)

        write_video(save_path, x, fps=fps, video_codec="h264", options={"crf": str(crf)})
    if verbose:
        print(f"Saved to {save_path}")
    return save_path


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def rand_size_crop_arr(pil_image, image_size):
    """
    Randomly crop image for height and width, ranging from image_size[0] to image_size[1]
    """
    arr = np.array(pil_image)

    # get random target h w
    height = random.randint(image_size[0], image_size[1])
    width = random.randint(image_size[0], image_size[1])
    # ensure that h w are factors of 8
    height = height - height % 8
    width = width - width % 8

    # get random start pos
    h_start = random.randint(0, max(len(arr) - height, 0))
    w_start = random.randint(0, max(len(arr[0]) - height, 0))

    # crop
    return Image.fromarray(arr[h_start : h_start + height, w_start : w_start + width])


def resize_crop_to_fill(pil_image, image_size):
    w, h = pil_image.size  # PIL is (W, H)
    th, tw = image_size
    rh, rw = th / h, tw / w
    if rh > rw:
        sh, sw = th, round(w * rh)
        image = pil_image.resize((sw, sh), Image.BICUBIC)
        i = 0
        j = int(round((sw - tw) / 2.0))
    else:
        sh, sw = round(h * rw), tw
        image = pil_image.resize((sw, sh), Image.BICUBIC)
        i = int(round((sh - th) / 2.0))
        j = 0
    arr = np.array(image)
    assert i + th <= arr.shape[0] and j + tw <= arr.shape[1]
    return Image.fromarray(arr[i : i + th, j : j + tw])


def map_target_fps(
    fps: float,
    max_fps: float,
) -> tuple[float, int]:
    """
    Map fps to a new fps that is less than max_fps.

    Args:
        fps (float): Original fps.
        max_fps (float): Maximum fps.

    Returns:
        tuple[float, int]: New fps and sampling interval.
    """
    if math.isnan(fps):
        return 0, 1
    if fps < max_fps:
        return fps, 1
    sampling_interval = math.ceil(fps / max_fps)
    new_fps = math.floor(fps / sampling_interval)
    return new_fps, sampling_interval


def sync_object_across_devices(obj: Any, rank: int = 0):
    """
    Synchronizes any picklable object across devices in a PyTorch distributed setting
    using `broadcast_object_list` with CUDA support.

    Parameters:
    obj (Any): The object to synchronize. Can be any picklable object (e.g., list, dict, custom class).
    rank (int): The rank of the device from which to broadcast the object state. Default is 0.

    Note: Ensure torch.distributed is initialized before using this function and CUDA is available.
    """

    # Move the object to a list for broadcasting
    object_list = [obj]

    # Broadcast the object list from the source rank to all other ranks
    dist.broadcast_object_list(object_list, src=rank, device="cuda")

    # Retrieve the synchronized object
    obj = object_list[0]

    return obj


def rescale_image_by_path(path: str, height: int, width: int):
    """
    Rescales an image to the specified height and width and saves it back to the original path.

    Args:
        path (str): The file path of the image.
        height (int): The target height of the image.
        width (int): The target width of the image.
    """
    try:
        # read image
        image = Image.open(path)

        # check if image is valid
        if image is None:
            raise ValueError("The image is invalid or empty.")

        # resize image
        resize_transform = transforms.Resize((width, height))
        resized_image = resize_transform(image)

        # save resized image back to the original path
        resized_image.save(path)

    except Exception as e:
        print(f"Error rescaling image: {e}")


def rescale_video_by_path(path: str, height: int, width: int):
    """
    Rescales an MP4 video (without audio) to the specified height and width.

    Args:
        path (str): The file path of the video.
        height (int): The target height of the video.
        width (int): The target width of the video.
    """
    try:
        # Read video and metadata
        video, info = read_video(path, backend="av")

        # Check if video is valid
        if video is None or video.size(0) == 0:
            raise ValueError("The video is invalid or empty.")

        # Resize video frames using a performant method
        resize_transform = transforms.Compose([transforms.Resize((width, height))])
        resized_video = torch.stack([resize_transform(frame) for frame in video])

        # Save resized video back to the original path
        resized_video = resized_video.permute(0, 2, 3, 1)
        write_video(path, resized_video, fps=int(info["video_fps"]), video_codec="h264")
    except Exception as e:
        print(f"Error rescaling video: {e}")


def save_tensor_to_disk(tensor, path, exist_handling="overwrite"):
    if os.path.exists(path):
        if exist_handling == "ignore":
            return
        elif exist_handling == "raise":
            raise UserWarning(f"File {path} already exists, rewriting!")
    torch.save(tensor, path)


def save_tensor_to_internet(tensor, path):
    raise NotImplementedError("save_tensor_to_internet is not implemented yet!")


def save_latent(latent, path, exist_handling="overwrite"):
    if path.startswith(("http://", "https://", "ftp://", "sftp://")):
        save_tensor_to_internet(latent, path)
    else:
        save_tensor_to_disk(latent, path, exist_handling=exist_handling)


def cache_latents(latents, path, exist_handling="overwrite"):
    for i in range(latents.shape[0]):
        save_latent(latents[i], path[i], exist_handling=exist_handling)
