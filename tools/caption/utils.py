import time

import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader

from tools.datasets.utils import extract_frames, is_video

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
PROMPTS = {
    "image": {
        "text": "Describe this image and its style to generate a succinct yet informative description. Pay attention to all objects in the image. The description should be useful for AI to re-generate the image. The description should be no more than five sentences. Remember do not exceed 5 sentences.",
        "type": "image",
    },
    "image-text": {
        "text": "Describe this image and its style in a very detailed manner. Pay attention to all objects in the image. The description should be useful for AI to re-generate the image. The description should be no more than six sentences. Some information about the image is '{}'.",
        "type": "image",
    },
    "image-3ex": {
        "text": "An image is given. Describe this image and its style to generate a succinct yet informative description. Pay attention to all objects in the image. The description should be useful for AI to re-generate the video. The description should be no more than five sentences. Here are some examples of good descriptions: 1. A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick and walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about. 2. Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field. 3. Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff's edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff’s edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway.",
        "type": "image",
    },
    "video": {
        "text": "Describe this video and its style in a very detailed manner. Pay attention to all objects in the video. The description should be useful for AI to re-generate the video. The description should be no more than six sentences.",
        "type": "video",
    },
    "video-text": {
        "text": "Describe this video and its style in a very detailed manner. Some information about the image is '{}'. Pay attention to all objects in the video. The description should be useful for AI to re-generate the video. The description should be no more than six sentences.",
        "type": "video",
    },
    "video-f1-detail-3ex": {
        "text": "A video is given by providing the middle frame. Describe this video and its style to generate a description. Pay attention to all objects in the video. The description should be useful for AI to re-generate the video. The description should be no more than six sentences. Here are some examples of good descriptions: 1. A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about. 2. Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field. 3. Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff's edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff’s edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway.",
        "type": "video",
    },
    "video-f1-detail-2ex-text": {
        "text": "A video is given by providing the middle frame. Some information about the image is '{}'. Describe this video and its style to generate a description. Pay attention to all objects in the video. Do not describe each frame individually. Do not reply with words like 'first frame'. The description should be useful for AI to re-generate the video. The description should be no more than six sentences. Here are some examples of good descriptions: 1. A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about. 2. Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field.",
        "type": "video",
    },
    "video-f3-detail-3ex": {
        "text": "A video is given by providing three frames in chronological order. Describe this video and its style to generate a description. Pay attention to all objects in the video. Do not describe each frame individually. Do not reply with words like 'first frame'. The description should be useful for AI to re-generate the video. The description should be no more than six sentences. Here are some examples of good descriptions: 1. A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about. 2. Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field. 3. Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff's edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff’s edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway.",
        "type": "video",
    },
    "video-f3-detail-2ex-text": {
        "text": "A video is given by providing three frames in chronological order. Some information about the image is '{}'. Describe this video and its style to generate a description. Pay attention to all objects in the video. Do not describe each frame individually. Do not reply with words like 'first frame'. The description should be useful for AI to re-generate the video. The description should be no more than six sentences. Here are some examples of good descriptions: 1. A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about. 2. Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field.",
        "type": "video",
    },
}


NUM_FRAMES_POINTS = {
    1: (0.5,),
    2: (0.25, 0.75),
    3: (0.1, 0.5, 0.9),
}


def read_file(input_path):
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path)
    elif input_path.endswith(".parquet"):
        return pd.read_parquet(input_path)
    else:
        raise NotImplementedError(f"Unsupported file format: {input_path}")


class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None, num_frames=3, get_text_input_ids=None, resize=None):
        self.csv_path = csv_path
        self.transform = transform
        self.data = read_file(csv_path)
        self.points = NUM_FRAMES_POINTS[num_frames]
        self.get_text_input_ids = get_text_input_ids
        self.use_text = False
        self.resize_size = resize
        self.resize = transforms.Resize(resize, transforms.InterpolationMode.BICUBIC) if resize is not None else None
        if "text" in self.data.columns:
            self.use_text = True

    def getitem(self, index):
        sample = self.data.iloc[index]
        path = sample["path"]
        if not is_video(path):
            images = [pil_loader(path)]
            length = 1
        else:
            images, length = extract_frames(sample["path"], points=self.points, backend="opencv", return_length=True)
        if self.resize_size is not None:
            images_r = []
            for img in images:
                if img.size[0] > self.resize_size or img.size[1] > self.resize_size:
                    img = self.resize(img)
                images_r.append(img)
            images = images_r
        imgs_size = [img.size for img in images]
        if self.transform is not None:
            images = self.transform(images)

        # we put images into a list as pytorch dataloader does not accept Pill
        out = dict(path=path, image=images, length=length, img_size=imgs_size)
        if self.get_text_input_ids is not None:
            if self.use_text:
                out["text"] = self.get_text_input_ids(sample["text"])
            else:
                out["text"] = self.get_text_input_ids()
        else:
            if self.use_text:
                out["text"] = sample["text"]
            else:
                out["text"] = ""
        return out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.getitem(index)


def collate_fn(batch):
    paths = [item["path"] for item in batch]
    images = [item["image"] for item in batch]
    lengths = [item["length"] for item in batch]
    img_sizes = [item["img_size"] for item in batch]
    texts = [item["text"] for item in batch]
    return paths, images, lengths, img_sizes, texts


class Timer:
    def __init__(self):
        self.time_taken = 0
        self.start_time = 0
        self.end_time = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.end_time = time.time()
        self.time_taken = self.end_time - self.start_time
