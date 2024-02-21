# 🎥 Open-Sora

## 📍 Overview

This repository is an unofficial implementation of OpenAI's Sora. We built this based on the [facebookresearch/DiT](https://github.com/facebookresearch/DiT) repository.

## Dataset preparation

We use [MSR-VTT](https://cove.thecvf.com/datasets/839) dataset, which is a large-scale video description dataset. We should preprocess the raw videos before training the model.

Before running `preprocess_data.py`, you should prepare a captions file and a video directory. The captions file should be a JSON file or a JSONL file. The video directory should contain all the videos.

Here is an example of the captions file:

```json
[
    {
        "file": "video0.mp4",
        "captions": ["a girl is throwing away folded clothes", "a girl throwing cloths around"]
    },
    {
        "file": "video1.mp4",
        "captions": ["a  comparison of two opposing team football athletes"]    
    }
]
```

Here is an example of the video directory:

```
.
├── video0.mp4
├── video1.mp4
└── ...
```

Each video may have multiple captions. So the outputs are video-caption pairs. E.g., the first video has two captions, then the output will be two video-caption pairs.

We use [VQ-VAE](https://github.com/wilson1yan/VideoGPT/) to quantize the video frames. And we use [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#clip) to extract the text features.

The output is an arrow dataset, which contains the following columns: "video_file", "video_latent_states", "text_latent_states". The dimension of "video_latent_states" is (T, H, W), and the dimension of "text_latent_states" is (S, D).

How to run the script:

```bash
python preprocess_data.py /path/to/captions.json /path/to/video_dir /path/to/output_dir
```

Note that this script needs to be run on a machine with a GPU. To avoid CUDA OOM, we filter out the videos that are too long.