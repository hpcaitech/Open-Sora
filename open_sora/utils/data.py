import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from colossalai.utils import get_current_device
from datasets import Dataset as HFDataset
from datasets import dataset_dict, load_from_disk
from diffusers.models import AutoencoderKL
from torch.utils.data import ConcatDataset, Dataset
from torchvision.io import read_video
from transformers import AutoModel

DatasetType = Union[Dataset, ConcatDataset, dataset_dict.Dataset]
PathType = Union[str, os.PathLike]


def ceil_to_multiple(x: int, multiple: int) -> int:
    m = x % multiple
    if m == 0:
        return x
    return x + multiple - m


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
            patch = patches[:, (y // patch_size) * num_x_patches + x // patch_size]
            video[:, :, y : y + patch_size, x : x + patch_size].copy_(patch)
    return video


def pad_sequences(
    sequences: List[torch.Tensor], pad_to_multiple: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of sequences.

    Args:
        sequences (List[torch.Tensor]): Each sequence is a tensor of shape [T, ...].

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Padded batch of sequences ([B, T, ...]) and padding mask ([B, T]).
    """
    max_len = max([sequence.shape[0] for sequence in sequences])
    if pad_to_multiple is not None:
        max_len = ceil_to_multiple(max_len, pad_to_multiple)
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
    videos: List[torch.Tensor], patch_size: int, pad_to_multiple: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Patchify a batch of videos.

    Args:
        videos (List[torch.Tensor]): A list of tensors of shape [T, C, H, W]
        patch_size (int): The size of the patches.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Padded batch of patches ([B, S, C, P, P]) and padding mask ([B, S]).
    """
    video_patches = [video2col(video, patch_size) for video in videos]
    return pad_sequences(video_patches, pad_to_multiple=pad_to_multiple)


def expand_mask_4d(q_mask: torch.Tensor, kv_mask: torch.Tensor) -> torch.Tensor:
    """expand 2D mask to 4D mask

    Args:
        q_mask (torch.Tensor): [B, Sq]
        kv_mask (torch.Tensor): [B, Skv]

    Returns:
        torch.Tensor: [B, 1, Sq, Skv]
    """
    q_len = q_mask.shape[1]
    mask = kv_mask.unsqueeze(1).repeat(1, q_len, 1)
    mask = mask * q_mask.unsqueeze(-1)
    return mask.unsqueeze(1)


def make_batch(
    samples: List[dict],
    video_dir: str,
    pad_to_multiple: Optional[int] = None,
    use_pooled_text: bool = False,
) -> dict:
    """Make a batch of samples.

    Args:
        samples (List[dict]): A list of samples.

    Returns:
        dict: A batch of samples.
    """
    videos = [
        read_video(os.path.join(video_dir, sample["video_file"]), pts_unit="sec")[0]
        for sample in samples
    ]
    texts = [sample["text_latent_states"] for sample in samples]
    if use_pooled_text:
        texts = torch.stack(texts, dim=0)
        return {
            "videos": videos,
            "text_latent_states": texts,
        }
    texts, text_padding_mask = pad_sequences(texts, pad_to_multiple=pad_to_multiple)
    return {
        "videos": videos,
        "text_latent_states": texts,
        "text_padding_mask": text_padding_mask,
    }


def normalize_video(video: torch.Tensor) -> torch.Tensor:
    return video.float() / 255 - 0.5


def unnormalize_video(video: torch.Tensor) -> torch.Tensor:
    return (video + 0.5) * 255


class VideoCompressor:
    t_factor: int
    h_w_factor: int
    out_channels: int

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Encode a video.

        Args:
            video (torch.Tensor): [T, H, W, C]

        Returns:
            torch.Tensor: [T, C, H, W]
        """
        raise NotImplementedError

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent tensor.

        Args:
            latent (torch.Tensor): [T, C, H, W]

        Returns:
            torch.Tensor: [T, H, W, C]
        """
        raise NotImplementedError


class RawVideoCompressor(VideoCompressor):
    t_factor = 1
    h_w_factor = 1
    out_channels = 3

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        # [T, H, W, C] -> [T, C, H, W]
        return video.permute(0, 3, 1, 2).contiguous()

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        # [T, C, H, W] -> [T, H, W, C]
        return latent.permute(0, 2, 3, 1).contiguous()


class VqvaeVideoCompressor(VideoCompressor):
    t_factor = 2
    h_w_factor = 4

    def __init__(self, vqvae: nn.Module):
        self.vqvae = vqvae
        self.out_channels = vqvae.embedding_dim

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        # [T, H, W, C] -> [B, C, T, H, W]
        video = video.permute(3, 0, 1, 2).unsqueeze(0)
        latent_indices, embeddings = self.vqvae.encode(video, include_embeddings=True)
        # [B, C, T, H, W] -> [T, C, H, W]
        return embeddings.squeeze(0).permute(1, 0, 2, 3)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        # [T, C, H, W] -> [B, C, T, H, W]
        latent = latent.permute(1, 0, 2, 3).unsqueeze(0)
        video = self.vqvae.decode_from_embeddings(latent)
        # [B, C, T, H, W] -> [T, H, W, C]
        video = video.squeeze(0).permute(1, 2, 3, 0)
        return video


class VaeVideoCompressor(VideoCompressor):
    t_factor = 1
    h_w_factor = 8
    out_channels = 4

    def __init__(self, vae: nn.Module):
        self.vae = vae

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        # [T, H, W, C] -> [T, C, H, W]
        video = video.permute(0, 3, 1, 2)
        return self.vae.encode(video).latent_dist.sample().mul_(0.18215)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        video = self.vae.decode(latent / 0.18215).sample
        # [T, C, H, W] -> [T, H, W, C]
        return video.permute(0, 2, 3, 1).contiguous()


def create_video_compressor(
    compressor_type: str,
    vqvae_path="hpcai-tech/vqvae",
    vae_path="stabilityai/sd-vae-ft-mse",
) -> VideoCompressor:
    if compressor_type == "raw":
        return RawVideoCompressor()
    if compressor_type == "vqvae":
        vqvae = (
            AutoModel.from_pretrained(vqvae_path, trust_remote_code=True)
            .to(get_current_device())
            .eval()
        )
        return VqvaeVideoCompressor(vqvae)
    if compressor_type == "vae":
        vae = AutoencoderKL.from_pretrained(vae_path).to(get_current_device()).eval()
        return VaeVideoCompressor(vae)
    raise ValueError(f"Unsupported video compressor type {compressor_type}")


@torch.no_grad()
def preprocess_batch(
    batch: dict,
    patch_size: int,
    video_compressor: VideoCompressor,
    device=None,
    pad_to_multiple: Optional[int] = None,
    model_arch: str = "cross-attn",
) -> dict:
    if device is None:
        device = get_current_device()
    videos = []
    for video in batch.pop("videos"):
        video = video.to(device)
        video = normalize_video(video)
        video = video_compressor.encode(video)
        videos.append(video)
    video_latent_states, video_padding_mask = patchify_batch(
        videos, patch_size, pad_to_multiple
    )
    batch["video_latent_states"] = video_latent_states
    batch["video_padding_mask"] = video_padding_mask
    if model_arch == "adaln":
        batch["attention_mask"] = expand_mask_4d(video_padding_mask, video_padding_mask)
    else:
        text_padding_mask = batch.pop("text_padding_mask").to(device)
        if model_arch == "cross-attn":
            batch["attention_mask"] = expand_mask_4d(
                video_padding_mask, text_padding_mask
            )
        else:
            attention_mask = torch.cat([text_padding_mask, video_padding_mask], dim=1)
            batch["attention_mask"] = expand_mask_4d(attention_mask, attention_mask)
    batch["text_latent_states"] = batch["text_latent_states"].to(device)
    return batch


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
