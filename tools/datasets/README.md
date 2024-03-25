# Dataset Download and Management

## HD-VG-130M

This dataset comprises 130M text-video pairs. You can download the dataset and prepare it for training according to [the dataset repository's instructions](https://github.com/daooshee/HD-VG-130M). There is a README.md file in the Google Drive link that provides instructions on how to download and cut the videos. For this version, we directly use the dataset provided by the authors.

## VidProM

```bash
python -m tools.datasets.convert_dataset vidprom VIDPROM_FOLDER --info VidProM_semantic_unique.csv
```

## Demo Dataset

You can use ImageNet and UCF101 for a quick demo. After downloading the datasets, you can use the following command to prepare the csv file for the dataset:

```bash
# ImageNet
python -m tools.datasets.convert_dataset imagenet IMAGENET_FOLDER --split train
# UCF101
python -m tools.datasets.convert_dataset ucf101 UCF101_FOLDER --split videos
```

## Dataset Format

The dataset should be provided in a CSV file, which is used both for training and data preprocessing. The CSV file should only contain the following columns (can be optional):

```csv
path, text, num_frames, aesthetic_score, fps, width, height, aspect_ratio
/absolute/path/to/image1.jpg, caption1, num_of_frames, score1
/absolute/path/to/video2.mp4, caption2, num_of_frames, score2
```

We use pandas to manage the CSV files. You can use the following code to read and write the CSV files:

```python
df = pd.read_csv(input_path)
df = df.to_csv(output_path, index=False)
```

## Manage datasets

We provide `csvutils.py` to manage the CSV files. You can use the following commands to process the CSV files:

```bash
# csvutil takes multiple CSV files as input and merge them into one CSV file
python -m tools.datasets.csvutil DATA1.csv DATA2.csv
# filter frames between 128 and 256, with captions
python -m tools.datasets.csvutil DATA.csv --fmin 128 --fmax 256 --remove-empty-caption
# compute the number of frames for each video
python -m tools.datasets.csvutil DATA.csv --relength
# remove caption prefix
python -m tools.datasets.csvutil DATA.csv --remove-caption-prefix
# generate DATA_root.csv with absolute path
python -m tools.datasets.csvutil DATA.csv --abspath /absolute/path/to/dataset
```
