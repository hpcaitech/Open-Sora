# Scoring and Filtering

- [Scoring and Filtering](#scoring-and-filtering)
  - [Aesthetic Score](#aesthetic-score)
  - [Optical Flow Score](#optical-flow-score)
  - [OCR](#ocr)
  - [Matching Score](#matching-score)
  - [Filtering](#filtering)

## Aesthetic Score

To evaluate the aesthetic quality of videos, we use the scoring model from [CLIP+MLP Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor). This model is trained on 176K SAC (Simulacra Aesthetic Captions) pairs, 15K LAION-Logos (Logos) pairs, and 250K AVA (The Aesthetic Visual Analysis) image-text pairs.

The aesthetic score is between 1 and 10, where 5.5 can be considered as the threshold for fair aesthetics, and 6.5 for high aesthetics. Good text-to-image models can achieve a score of 7.0 or higher.

For videos, we extract the first, last, and the middle frames for evaluation. The script also supports images as input.
The throughput of our code is ~1K videos/s on a single H800 GPU. It also supports running on multiple GPUs for further acceleration.

First, install the required packages following our [installation instructions](../../docs/installation.md)'s "Data Dependencies".

Next, download the scoring model to `./pretrained_models/aesthetic.pth`.

```bash
wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth -O pretrained_models/aesthetic.pth
```

<!-- First, install the required packages and download the scoring model to `./pretrained_models/aesthetic.pth`.
```bash
# pip install
pip install git+https://github.com/openai/CLIP.git
pip install decord

# get pretrained model
wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth -O pretrained_models/aesthetic.pth
``` -->

Then, run the following command. **Make sure** the meta file has column `path` (path to the sample).
```bash
torchrun --nproc_per_node 8 -m tools.scoring.aesthetic.inference /path/to/meta.csv --bs 1024 --num_workers 16
```
This will generate multiple part files, each corresponding to a node . Run `python -m tools.datasets.datautil /path/to/meta_aes_part*.csv --output /path/to/meta_aes.csv` to merge them.

## Optical Flow Score

Optical flow scores are used to assess the motion of a video. Higher optical flow scores indicate larger movement.
We use the [UniMatch](https://github.com/autonomousvision/unimatch) model for this task.

First, install the required packages following our [installation instructions](../../docs/installation.md)'s "Data Dependencies".

Next, download the pretrained model to `./pretrained_model/unimatch/`
```bash
wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth -P ./pretrained_models/unimatch/
```

Then, run the following command. **Make sure** the meta file has column `path` (path to the sample).
```bash
torchrun --standalone --nproc_per_node 8 tools/scoring/optical_flow/inference.py /path/to/meta.csv
```

This should output `/path/to/meta_flow.csv` with column `flow`.

## OCR
Some videos are of dense text scenes like news broadcast and advertisement, which are not desired for training.
We apply Optical Character Recognition (OCR) to detect texts and drop samples with dense texts. Here, we use
the [DBNet++](https://arxiv.org/abs/2202.10304) model implemented by [MMOCR](https://github.com/open-mmlab/mmocr/).

First, install the required packages following our [installation instructions](../../docs/installation.md)'s "Data Dependencies" and "OCR" section.

<!-- First, install [MMOCR](https://mmocr.readthedocs.io/en/dev-1.x/get_started/install.html).
For reference, we install packages of these versions.
```
torch==2.0.1
mmcv==2.0.1
mmdet==3.1.0
mmocr==1.0.1
``` -->

Then, run the following command. **Make sure** the meta file has column `path` (path to the sample).
<!-- ```bash
torchrun --standalone --nproc_per_node 8 tools/scoring/ocr/inference.py /path/to/meta.csv
``` -->
```bash
torchrun --standalone --nproc_per_node 8 -m tools.scoring.ocr.inference /path/to/meta.csv
```
This should output `/path/to/meta_ocr.csv` with column `ocr`, indicating the number of text regions with detection confidence > 0.3.


## Matching Score

Matching scores are calculated to evaluate the alignment between an image/video and its caption.
Here, we use the [CLIP](https://github.com/openai/CLIP) model, which is trained on image-text pairs.
We simply use the cosine similarity as the matching score.
For videos, we extract the middle frame and compare it with the caption.

First, install OpenAI CLIP.
```bash
pip install git+https://github.com/openai/CLIP.git
```

Then, run the following command. **Make sure** the meta file has column `path` (path to the sample) and `text` (caption of the sample).

```bash
torchrun --standalone --nproc_per_node 8 tools/scoring/matching/inference.py /path/to/meta.csv
```

This should output `/path/to/meta_match.csv` with column `match`. Higher matching scores indicate better image-text/video-text alignment.


## Filtering
Once scores are obtained, it is simple to filter samples based on these scores. Here is an example to remove
samples of aesthetic score < 5.0.
```
python -m tools.datasets.datautil /path/to/meta.csv --aesmin 5.0
```
This should output `/path/to/meta_aesmin5.0.csv` with column `aes` >= 5.0
