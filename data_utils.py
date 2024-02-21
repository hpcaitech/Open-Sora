import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from datasets import dataset_dict, load_from_disk
from torch.utils.data import ConcatDataset, Dataset

DatasetType = Union[Dataset, ConcatDataset, dataset_dict.Dataset]
PathType = Union[str, os.PathLike]


def video2col(video_4d: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert a 4D video tensor to a 2D tensor where each row is a patch of the video.

    Args:
        video_4d (torch.Tensor): A tensor of shape [T, C, H, W]
        patch_size (int): The size of the patches.

    Returns:
        torch.Tensor: A tensor of shape [S, C, P, P] where S is the number of patches and P is the patch size.
    """
    t, c, h, w = video_4d.shape
    out = []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            if y + patch_size > h or x + patch_size > w:
                continue
            patch = video_4d[:, :, y : y + patch_size, x : x + patch_size]
            out.append(patch)
    # [S, C, P, P]
    return torch.stack(out, dim=1).view(-1, c, patch_size, patch_size)


def col2video(
    patches: torch.Tensor, video_shape: Tuple[int, int, int, int]
) -> torch.Tensor:
    """
    Convert a 2D tensor of patches to a 4D video tensor.

    Args:
        patches (torch.Tensor): A tensor of shape [S, C, P, P] where S is the number of patches and P is the patch size.
        video_shape (Tuple[int, int, int, int]): The shape of the video tensor [T, C, H, W].

    Returns:
        torch.Tensor: A tensor of shape [T, C, H, W].
    """
    t, c, h, w = video_shape
    video = torch.empty(t, c, h, w, dtype=patches.dtype, device=patches.device)
    patch_size = patches.shape[2]
    num_x_patches = w // patch_size
    patches = patches.view(t, -1, c, patch_size, patch_size)
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            if y + patch_size > h or x + patch_size > w:
                continue
            # [T, C, P, P]
            patch = patches[:, y * num_x_patches + x]
            video[:, :, y : y + patch_size, x : x + patch_size].copy_(patch)
    return video


def pad_sequences(sequences: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of sequences.

    Args:
        sequences (List[torch.Tensor]): Each sequence is a tensor of shape [T, ...].

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Padded batch of sequences ([B, T, ...]) and padding mask ([B, T]).
    """
    max_len = max([sequence.shape[0] for sequence in sequences])
    padded_sequences = [
        F.pad(
            sequence, [0] * (sequence.ndim - 1) * 2 + [0, max_len - sequence.shape[0]]
        )
        for sequence in sequences
    ]
    padded_sequences = torch.stack(padded_sequences, dim=0)
    padding_mask = torch.zeros(
        padded_sequences.shape[0],
        padded_sequences.shape[1],
        dtype=torch.int,
        device=padded_sequences.device,
    )
    for i, sequence in enumerate(sequences):
        padding_mask[i, : sequence.shape[0]] = 1
    return padded_sequences, padding_mask


def patchify_batch(
    videos: List[torch.Tensor], patch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Patchify a batch of videos.

    Args:
        videos (List[torch.Tensor]): A list of tensors of shape [T, C, H, W]
        patch_size (int): The size of the patches.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Padded batch of patches ([B, S, C, P, P]) and padding mask ([B, S]).
    """
    video_patches = [video2col(video, patch_size) for video in videos]
    return pad_sequences(video_patches)


def make_batch(samples: List[dict], patch_size: int) -> dict:
    """Make a batch of samples.

    Args:
        samples (List[dict]): A list of samples.

    Returns:
        dict: A batch of samples.
    """
    videos = [sample["video_latent_states"] for sample in samples]
    videos, video_padding_mask = patchify_batch(videos, patch_size)
    texts = [sample["text_latent_states"] for sample in samples]
    texts, text_padding_mask = pad_sequences(texts)
    return {
        "video_latent_states": videos,
        "video_padding_mask": video_padding_mask,
        "text_latent_states": texts,
        "text_padding_mask": text_padding_mask,
    }


def load_datasets(
    dataset_paths: Union[PathType, List[PathType]], mode: str = "train"
) -> Optional[DatasetType]:
    """
    Load pre-tokenized dataset.
    Each instance of dataset is a dictionary with
    `{'input_ids': List[int], 'labels': List[int], sequence: str}` format.
    """
    mode_map = {"train": "train", "dev": "validation", "test": "test"}
    assert mode in tuple(
        mode_map
    ), f"Unsupported mode {mode}, it must be in {tuple(mode_map)}"

    if isinstance(dataset_paths, (str, os.PathLike)):
        dataset_paths = [dataset_paths]

    datasets = []  # `List[datasets.dataset_dict.Dataset]`
    for ds_path in dataset_paths:
        ds_path = os.path.abspath(ds_path)
        assert os.path.exists(ds_path), f"Not existed file path {ds_path}"
        ds_dict = load_from_disk(
            dataset_path=ds_path, keep_in_memory=False
        ).with_format("torch")
        if isinstance(ds_dict, HFDataset):
            datasets.append(ds_dict)
        else:
            if mode_map[mode] in ds_dict:
                datasets.append(ds_dict[mode_map[mode]])
    if len(datasets) == 0:
        return None
    if len(datasets) == 1:
        return datasets.pop()
    return ConcatDataset(datasets=datasets)
