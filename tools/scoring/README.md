# Data Scoring and Filtering
Important!!! All scoring jobs require these columns in meta files:
- `path`: absolute path to a sample

## Aesthetic Score
First prepare the environment and pretrained models.
```bash
# install clip
pip install git+https://github.com/openai/CLIP.git
pip install decord

# get pretrained model
wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth -O pretrained_models/aesthetic.pth
```

Then run:
```bash
# output: DATA_aes.csv
python -m tools.scoring.aesthetic.inference /path/to/meta.csv
```
The output should be `/path/to/meta_aes.csv` with column `aes`. Aesthetic scores range from 1 to 10, with 10 being the best quality.

## Optical Flow Score
First get the pretrained model.
```bash
wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth -P pretrained_models/unimatch
```

Then run:
```
python tools/scoring/optical_flow/inference.py /path/to/meta.csv
```
The output should be `/path/to/meta_flow.csv` with column `flow`. Higher optical flow scores indicate larger movement.

## Matching Score
Require column `text` in meta files, which is the caption of the sample.

TODO.
