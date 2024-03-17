# Dataset Download and Management

## Dataset Format

The training data should be provided in a CSV file with the following format:

```csv
/absolute/path/to/image1.jpg, caption1, num_of_frames
/absolute/path/to/image2.jpg, caption2, num_of_frames
```

## HD-VG-130M

This dataset comprises 130M text-video pairs. You can download the dataset and prepare it for training according to [the dataset repository's instructions](https://github.com/daooshee/HD-VG-130M). There is a README.md file in the Google Drive link that provides instructions on how to download and cut the videos. For this version, we directly use the dataset provided by the authors.

## Demo Dataset

You can use ImageNet and UCF101 for a quick demo. After downloading the datasets, you can use the following command to prepare the csv file for the dataset:

```bash
# ImageNet
python -m tools.datasets.convert_dataset imagenet IMAGENET_FOLDER --split train
# UCF101
python -m tools.datasets.convert_dataset ucf101 UCF101_FOLDER --split videos
```

## Manage datasets

We provide `csvutils.py` to manage the CSV files. You can use the following commands to process the CSV files:

```bash
# generate DATA_fmin_128_fmax_256.csv with frames between 128 and 256
python -m tools.datasets.csvutil DATA.csv --fmin 128 --fmax 256
# generate DATA_root.csv with absolute path
python -m tools.datasets.csvutil DATA.csv --root /absolute/path/to/dataset
# remove videos with no captions
python -m tools.datasets.csvutil DATA.csv --remove-empty-caption
# compute the number of frames for each video
python -m tools.datasets.csvutil DATA.csv --relength
# remove caption prefix
python -m tools.datasets.csvutil DATA.csv --remove-caption-prefix
```

To merge multiple CSV files, you can use the following command:

```bash
cat *csv > combined.csv
```
