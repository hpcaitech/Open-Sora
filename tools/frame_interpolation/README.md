# Frame Interpolation

For current version, we sample 1 frame out of 3 frames in the video. Although we are going to use VAE to avoid frame loss, we provide a frame interpolation tool to interpolate the video now. The frame interpolation tool is based on [AMT](https://github.com/MCG-NKU/AMT).

Interpolation can be useful for scenery videos, but it may not be suitable for videos with fast motion.

## Requirement

Install the required dependancies by following our [installation instructions](../../docs/installation.md)'s "Data Dependencies" and "Frame Interpolation" sections.

<!-- ```bash
conda install -c conda-forge opencv
pip install imageio
``` -->

## Model

We use **AMT** as our frame interpolation model. After sampling, you can use frame interpolation model to interpolate your video smoothly.

## Usage

The ckpt file will be automatically downloaded in user's `.cache` directory. You can use frame interpolation to your video file or a video folder.

1. Process a video file

```python
python -m tools.frame_interpolation.interpolation your_video.mp4
```

2. Process all video file in target directory

```python
python -m tools.frame_interpolation.interpolation your_video_dir --output_path samples/interpolation
```

The output video will be stored at `output_path` and its duration time is equal `the total number of frames after frame interpolation / the frame rate`

### Command Line Arguments

* `input`: Path of the input video. **Video path** or **Folder path(with --folder)**
* `--ckpt`: Pretrained model of [AMT](https://github.com/MCG-NKU/AMT). Default path: `~/.cache/amt-g.pth`.
* `--niter`: Iterations of interpolation. With $m$ input frames, `[N_ITER]` $=n$ corresponds to $2^n\times (m-1)+1$ output frames.
* `--fps`: Frame rate of the input video. (Default: 8)
* `--output_path`: **Folder Path** of the output video.
