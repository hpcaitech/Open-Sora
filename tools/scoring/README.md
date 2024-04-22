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
torchrun --nproc_per_node 8 -m tools.scoring.aesthetic.inference /path/to/meta.csv --bs 1024 --num_workers 16
```

This will generate multiple part files, you can use `python -m tools.datasets.csvutil DATA1.csv DATA2.csv` to merge these part files.

## Optical Flow Score

Optical flow scores are used to assess the motion of a video. Higher optical flow scores indicate larger movement.
TODO: acknowledge UniMatch.

First get the pretrained model.

```bash
wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth -P pretrained_models/unimatch
```

Then run:

```bash
torchrun --standalone --nproc_per_node 8 tools/scoring/optical_flow/inference.py /path/to/meta.csv
```

The output should be `/path/to/meta_flow.csv` with column `flow`.

## Matching Score

Matching scores are calculated to evaluate the alignment between an image/video and its caption.
For videos, we compute the matching score of the middle frame and the caption.

**Make sure** meta files contain the column `text`, which is the caption of the sample. Then run:

```bash
torchrun --standalone --nproc_per_node 8 tools/scoring/matching/inference.py /path/to/meta.csv
```

The output should be `/path/to/meta_match.csv` with column `match`. Higher matching scores indicate better image-text/video-text alignment.
