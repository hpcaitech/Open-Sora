# Data Scoring and Filtering

- [Data Scoring and Filtering](#data-scoring-and-filtering)
  - [Aesthetic Scoring](#aesthetic-scoring)
    - [Requirement](#requirement)
    - [Usage](#usage)
  - [Optical Flow Score](#optical-flow-score)
  - [Matching Score](#matching-score)

## Aesthetic Scoring

To evaluate the aesthetic quality of videos, we use a pretrained model from [CLIP+MLP Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor). This model is trained on 176K SAC (Simulacra Aesthetic Captions) pairs, 15K LAION-Logos (Logos) pairs, and 250K AVA (The Aesthetic Visual Analysis) image-text pairs.

The score is between 1 and 10, where 5.5 can be considered as the threshold for fair aesthetics, and 6.5 for good aesthetics. Good text-to-image models can achieve a score of 7.0 or higher.

For videos, we extract the first, last, and the middle frames for evaluation. The script also supports images. Our script enables 1k videos/s with one GPU. It also supports multiple GPUs to further accelerate the process.

### Requirement

```bash
# install clip
pip install git+https://github.com/openai/CLIP.git
pip install decord

# get pretrained model
wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth -O pretrained_models/aesthetic.pth
```

### Usage

With `meta.csv` containing the paths to the videos, run the following command:

```bash
# output: meta_aes.csv
python -m tools.scoring.aesthetic.inference meta.csv
```

## Optical Flow Score

First get the pretrained model.

```bash
wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth -P pretrained_models/unimatch
```

With `meta.csv` containing the paths to the videos, run the following command:

```bash
python -m tools.scoring.optical_flow.inference /path/to/meta.csv
```

The output should be `/path/to/meta_flow.csv` with column `flow`. Higher optical flow scores indicate larger movement.

## Matching Score

Require column `text` in meta files, which is the caption of the sample.

TODO.
