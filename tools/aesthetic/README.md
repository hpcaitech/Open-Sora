# Aesthetic Scoring

To evaluate the aesthetic quality of videos, we use a pretrained model from [CLIP+MLP Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor). This model is trained on 176K SAC (Simulacra Aesthetic Captions) pairs, 15K LAION-Logos (Logos) pairs, and 250K AVA (The Aesthetic Visual Analysis) image-text pairs.

The score is between 1 and 10, where 5.5 can be considered as the threshold for fair aesthetics, and 6.5 for good aesthetics. Good text-to-image models can achieve a score of 7.0 or higher.

For videos, we extract the first, last, and the middle frames for evaluation. The script also supports images. Our script enables 1k videos/s with one GPU. It also supports multiple GPUs to further accelerate the process.

## Requirement

```bash
# install clip
pip install git+https://github.com/openai/CLIP.git

# get pretrained model
wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth -O pretrained_models/aesthetic.pth
```

## Usage

With `DATA.csv` containing the paths to the videos, run the following command:

```bash
# output: DATA_aes.csv
python -m tools.aesthetic.inference DATA.csv
```
