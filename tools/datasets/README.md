# Dataset Management

- [Dataset Management](#dataset-management)
  - [Dataset Format](#dataset-format)
  - [Dataset to CSV](#dataset-to-csv)
  - [Manage datasets](#manage-datasets)
    - [Requirement](#requirement)
    - [Basic Usage](#basic-usage)
    - [Score filtering](#score-filtering)
    - [Documentation](#documentation)
  - [Transform datasets](#transform-datasets)
    - [Resize](#resize)
    - [Frame extraction](#frame-extraction)
    - [Crop Midjourney 4 grid](#crop-midjourney-4-grid)
  - [Analyze datasets](#analyze-datasets)

After preparing the raw dataset according to the [instructions](/docs/datasets.md), you can use the following commands to manage the dataset.

## Dataset Format

All dataset should be provided in a `.csv` file (or `parquet.gzip` to save space), which is used for both training and data preprocessing. The columns should follow the words below:

- `path`: the relative/absolute path or url to the image or video file. Required.
- `text`: the caption or description of the image or video. Required for training.
- `num_frames`: the number of frames in the video. Required for training.
- `width`: the width of the video frame. Required for dynamic bucket.
- `height`: the height of the video frame. Required for dynamic bucket.
- `aspect_ratio`: the aspect ratio of the video frame (height / width). Required for dynamic bucket.
- `resolution`: height x width. For analysis.
- `text_len`: the number of tokens in the text. For analysis.
- `aes`: aesthetic score calculated by [asethetic scorer](/tools/aesthetic/README.md). For filtering.
- `flow`: optical flow score calculated by [UniMatch](/tools/scoring/README.md). For filtering.
- `match`: matching score of a image-text/video-text pair calculated by [CLIP](/tools/scoring/README.md). For filtering.
- `fps`: the frame rate of the video. Optional.
- `cmotion`: the camera motion.

An example ready for training:

```csv
path, text, num_frames, width, height, aspect_ratio
/absolute/path/to/image1.jpg, caption, 1, 720, 1280, 0.5625
/absolute/path/to/video1.mp4, caption, 120, 720, 1280, 0.5625
/absolute/path/to/video2.mp4, caption, 20, 256, 256, 1
```

We use pandas to manage the `.csv` or `.parquet` files. The following code is for reading and writing files:

```python
df = pd.read_csv(input_path)
df = df.to_csv(output_path, index=False)
# or use parquet, which is smaller
df = pd.read_parquet(input_path)
df = df.to_parquet(output_path, index=False)
```

## Dataset to CSV

As a start point, `convert.py` is used to convert the dataset to a CSV file. You can use the following commands to convert the dataset to a CSV file:

```bash
python -m tools.datasets.convert DATASET-TYPE DATA_FOLDER

# general video folder
python -m tools.datasets.convert video VIDEO_FOLDER --output video.csv
# general image folder
python -m tools.datasets.convert image IMAGE_FOLDER --output image.csv
# imagenet
python -m tools.datasets.convert imagenet IMAGENET_FOLDER --split train
# ucf101
python -m tools.datasets.convert ucf101 UCF101_FOLDER --split videos
# vidprom
python -m tools.datasets.convert vidprom VIDPROM_FOLDER --info VidProM_semantic_unique.csv
```

## Manage datasets

Use `csvutil` to manage the dataset.

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

### Basic Usage

You can use the following commands to process the `csv` or `parquet` files. The output file will be saved in the same directory as the input, with different suffixes indicating the processed method.

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

# Remove corrupted video from the csv
python -m tools.datasets.csvutil DATA.csv --remove-corrupted

# Compute num_frames, height, width, fps, aspect_ratio for videos or images
# output: IMG_DATA+VID_DATA_vinfo.csv
python -m tools.datasets.csvutil IMG_DATA.csv VID_DATA.csv --video-info

# You can run multiple operations at the same time.
python -m tools.datasets.csvutil DATA.csv --video-info --remove-empty-caption --remove-url --lang en
```

### Score filtering

To examine and filter the quality of the dataset by aesthetic score and clip score, you can use the following commands:

```bash
# sort the dataset by aesthetic score
# output: DATA_sort.csv
python -m tools.datasets.csvutil DATA.csv --sort aesthetic_score
# View examples of high aesthetic score
head -n 10 DATA_sort.csv
# View examples of low aesthetic score
tail -n 10 DATA_sort.csv

# sort the dataset by clip score
# output: DATA_sort.csv
python -m tools.datasets.csvutil DATA.csv --sort clip_score

# filter the dataset by aesthetic score
# output: DATA_aesmin_0.5.csv
python -m tools.datasets.csvutil DATA.csv --aesmin 0.5
# filter the dataset by clip score
# output: DATA_matchmin_0.5.csv
python -m tools.datasets.csvutil DATA.csv --matchmin 0.5
```

### Documentation

You can also use `python -m tools.datasets.csvutil --help` to see usage.

| Args                        | File suffix    | Description                                                   |
| --------------------------- | -------------- | ------------------------------------------------------------- |
| `--output OUTPUT`           |                | Output path                                                   |
| `--format FORMAT`           |                | Output format (csv, parquet, parquet.gzip)                    |
| `--disable-parallel`        |                | Disable `pandarallel`                                         |
| `--seed SEED`               |                | Random seed                                                   |
| `--shard SHARD`             | `_0`,`_1`, ... | Shard the dataset                                             |
| `--sort KEY`                | `_sort`        | Sort the dataset by KEY                                       |
| `--sort-descending KEY`     | `_sort`        | Sort the dataset by KEY in descending order                   |
| `--difference DATA.csv`     |                | Remove the paths in DATA.csv from the dataset                 |
| `--intersection DATA.csv`   |                | Keep the paths in DATA.csv from the dataset and merge columns |
| `--info`                    | `_info`        | Get the basic information of each video and image             |
| `--ext`                     | `_ext`         | Remove rows if the file does not exist                        |
| `--remove-corrupted`        | `_nocorrupted` | Remove the corrupted video and image                          |
| `--relpath`                 | `_relpath`     | Modify the path to relative path by root given                |
| `--abspath`                 | `_abspath`     | Modify the path to absolute path by root given                |
| `--remove-empty-caption`    | `_noempty`     | Remove rows with empty caption                                |
| `--remove-url`              | `_nourl`       | Remove rows with url in caption                               |
| `--lang LANG`               | `_lang`        | Remove rows with other language                               |
| `--remove-path-duplication` | `_noduppath`   | Remove rows with duplicated path                              |
| `--remove-text-duplication` | `_noduptext`   | Remove rows with duplicated caption                           |
| `--refine-llm-caption`      | `_llm`         | Modify the caption generated by LLM                           |
| `--clean-caption MODEL`     | `_clean`       | Modify the caption according to T5 pipeline to suit training  |
| `--unescape`                | `_unescape`    | Unescape the caption                                          |
| `--merge-cmotion`           | `_cmotion`     | Merge the camera motion to the caption                        |
| `--count-num-token`         | `_ntoken`      | Count the number of tokens in the caption                     |
| `--load-caption EXT`        | `_load`        | Load the caption from the file                                |
| `--fmin FMIN`               | `_fmin`        | Filter the dataset by minimum number of frames                |
| `--fmax FMAX`               | `_fmax`        | Filter the dataset by maximum number of frames                |
| `--hwmax HWMAX`             | `_hwmax`       | Filter the dataset by maximum height x width                  |
| `--aesmin AESMIN`           | `_aesmin`      | Filter the dataset by minimum aesthetic score                 |
| `--matchmin MATCHMIN`       | `_matchmin`    | Filter the dataset by minimum clip score                      |
| `--flowmin FLOWMIN`         | `_flowmin`     | Filter the dataset by minimum optical flow score              |

## Transform datasets

The `tools.datasets.transform` module provides a set of tools to transform the dataset. The general usage is as follows:

```bash
python -m tools.datasets.transform TRANSFORM_TYPE META.csv ORIGINAL_DATA_FOLDER DATA_FOLDER_TO_SAVE_RESULTS --additional-args
```

### Resize

Sometimes you may need to resize the images or videos to a specific resolution. You can use the following commands to resize the dataset:

```bash
python -m tools.datasets.transform meta.csv /path/to/raw/data /path/to/new/data --length 2160
```

### Frame extraction

To extract frames from videos, you can use the following commands:

```bash
python -m tools.datasets.transform vid_frame_extract meta.csv /path/to/raw/data /path/to/new/data --points 0.1 0.5 0.9
```

### Crop Midjourney 4 grid

Randomly select one of the 4 images in the 4 grid generated by Midjourney.

```bash
python -m tools.datasets.transform img_rand_crop meta.csv /path/to/raw/data /path/to/new/data
```

## Analyze datasets

You can easily get basic information about a `.csv` dataset by using the following commands:

```bash
# examine the first 10 rows of the CSV file
head -n 10 DATA1.csv
# count the number of data in the CSV file (approximately)
wc -l DATA1.csv
```

For the dataset provided in a `.csv` or `.parquet` file, you can easily analyze the dataset using the following commands. Plots will be automatically saved.

```python
pyhton -m tools.datasets.analyze DATA_info.csv
```
