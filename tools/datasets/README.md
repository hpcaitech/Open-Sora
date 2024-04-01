# Dataset Management

After preparing the raw dataset according to the [instructions](/docs/datasets.md), you can use the following commands to manage the dataset.

## Dataset Format

All dataset should be provided in a CSV file, which is used both for training and data preprocessing.. The CSV file should only contain the following columns (can be optional).

```csv
path, text, num_frames, fps, width, height, aspect_ratio, aesthetic_score, clip_score, ...
/absolute/path/to/image1.jpg, caption1, num_of_frames
/absolute/path/to/video2.mp4, caption2, num_of_frames
```

We use pandas to manage the CSV files. The following code is for reading and writing the CSV files:

```python
df = pd.read_csv(input_path)
df = df.to_csv(output_path, index=False)
```

The columns are defined as follows:

- `path`: the relative/absolute path or url to the image or video file. The only required column.
- `text`: the caption or description of the image or video. Necessary for training.
- `num_frames`: the number of frames in the video. Necessary for training.
- `fps`: the frame rate of the video. Optional.
- `width`: the width of the video frame. Necessary for STDiT2.
- `height`: the height of the video frame. Necessary for STDiT2.
- `aspect_ratio`: the aspect ratio of the video frame (height divided by width). Optional.
- `aesthetic_score`: the aesthetic score by [asethetic scorer](/tools/aesthetic/README.md). Optional.
- `clip_score`: the clip score by [clip scorer](/tools/clip/README.md). Optional.

## Dataset to CSV

As a start point, `convert.py` is used to convert the dataset to a CSV file. You can use the following commands to convert the dataset to a CSV file:

```bash
python -m tools.datasets.convert DATASET-TYPE DATA_FOLDER
# general video folder
python -m tools.datasets.convert video VIDEO_FOLDER
# general image folder
python -m tools.datasets.convert image IMAGE_FOLDER
# imagenet
python -m tools.datasets.convert imagenet IMAGENET_FOLDER --split train
# ucf101
python -m tools.datasets.convert ucf101 UCF101_FOLDER --split videos
# vidprom
python -m tools.datasets.convert vidprom VIDPROM_FOLDER --info VidProM_semantic_unique.csv
```

## Manage datasets

You can easily get basic information about the dataset by using the following commands:

```bash
# examine the first 10 rows of the CSV file
head -n 10 DATA1.csv
# count the number of data in the CSV file (approximately)
wc -l DATA1.csv
```

Additionally, Ww provide `csvutils.py` to manage the CSV files.

### Requirement

To accelerate processing speed, you can install [pandarallel](https://github.com/nalepae/pandarallel):

```bash
pip install pandarallel
```

To get video information, you need to install [opencv-python](https://github.com/opencv/opencv-python):

```bash
pip install opencv-python
```

To filter a specific language, you need to install [lingua](https://github.com/pemistahl/lingua-py):

```bash
pip install lingua-language-detector
```

### Usage

You can use the following commands to process the CSV files. The output csv file will be saved in the same directory as the input csv file, with different suffixes indicating the processing method.

```bash
# csvutil takes multiple CSV files as input and merge them into one CSV file
# output: DATA1+DATA2.csv
python -m tools.datasets.csvutil DATA1.csv DATA2.csv
# shard CSV files into multiple CSV files
# output: DATA1_0.csv, DATA1_1.csv, ...
python -m tools.datasets.csvutil DATA1.csv --shard 10
# filter frames between 128 and 256, with captions
# output: DATA1_fmin_128_fmax_256.csv
python -m tools.datasets.csvutil DATA.csv --fmin 128 --fmax 256
# Disable parallel processing
python -m tools.datasets.csvutil DATA.csv --fmin 128 --fmax 256 --disable-parallel
```

Here are more examples:

```bash
# modify the path to absolute path by root given
# output: DATA_abspath.csv
python -m tools.datasets.csvutil DATA.csv --abspath /absolute/path/to/dataset
# modify the path to relative path by root given
# output: DATA_relpath.csv
python -m tools.datasets.csvutil DATA.csv --relpath /relative/path/to/dataset

# remove the rows with empty captions
# output: DATA_noempty.csv
python -m tools.datasets.csvutil DATA.csv --remove-empty-caption
# remove the rows with urls
# output: DATA_nourl.csv
python -m tools.datasets.csvutil DATA.csv --remove-url
# unescape the caption
# output: DATA_unescape.csv
python -m tools.datasets.csvutil DATA.csv --unescape
# modify LLaVA caption
# output: DATA_rcp.csv
python -m tools.datasets.csvutil DATA.csv --remove-caption-prefix
# keep only the rows with english captions
# output: DATA_en.csv
python -m tools.datasets.csvutil DATA.csv --lang en

# compute num_frames, height, width, fps, aspect_ratio for videos or images
# output: IMG_DATA+VID_DATA_vinfo.csv
python -m tools.datasets.csvutil IMG_DATA.csv VID_DATA --video-info
```

You can apply multiple operations at the same time:

```bash
# output: DATA_vinfo_noempty_nourl_en.csv
python -m tools.datasets.csvutil DATA.csv --video-info --remove-empty-caption --remove-url --lang en
```

To examine and filter the quality of the dataset by aesthetic score and clip score, you can use the following commands:

```bash
# sort the dataset by aesthetic score
# output: DATA_sort.csv
python -m tools.datasets.csvutil DATA.csv --sort-descending aesthetic_score
# View examples of high aesthetic score
head -n 10 DATA_sort.csv
# View examples of low aesthetic score
tail -n 10 DATA_sort.csv

# sort the dataset by clip score
# output: DATA_sort.csv
python -m tools.datasets.csvutil DATA.csv --sort-descending clip_score

# filter the dataset by aesthetic score
# output: DATA_aesmin_0.5.csv
python -m tools.datasets.csvutil DATA.csv --aesmin 0.5
# filter the dataset by clip score
# output: DATA_matchmin_0.5.csv
python -m tools.datasets.csvutil DATA.csv --matchmin 0.5
```

## Frame extraction speed

We use three libraries to extract frames from videos: `opencv`, `pyav` and `decord`. Our benchmark results of loading 256 video's middle frames are as follows:

| Library | Time (s) |
| ------- | -------- |
| opencv  | 33       |
| decord  | 28       |
| pyav    | 10       |

Although `pyav` is the fastest, it can only extract the key frames instead of frames at any time. Therefore, we use `decord` as the default library for frame extraction. For dataset management, without a bottleneck on loading speed, we choose `opencv` as the default library for video information extraction.
