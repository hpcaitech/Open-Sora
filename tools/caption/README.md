# Video Captioning

Human labeling of videos is expensive and time-consuming. We adopt powerful image captioning models to generate captions for videos. Although GPT-4V achieves a better performance, its 20s/sample speed is too slow for us. As for our v1.2 model, we captioned our training videos with the [PLLaVA](https://github.com/magic-research/PLLaVA) model. PLLaVA performs highly competitively on multiple video-based text generation benchmarks including [MVbench](https://paperswithcode.com/sota/video-question-answering-on-mvbench?p=pllava-parameter-free-llava-extension-from-1).

## PLLaVA Captioning

To balance captioning speed and performance, we chose the 13B version of PLLaVA configured with 2*2 spatial pooling. We feed it with 4 frames evenly extracted from the video. We accelerate its inference via (1) batching and (2) offload frame extraction to a separate process such that the GPU computations and frame extraction happen in parallel.

### Installation
Install the required dependancies by following our [installation instructions](../../docs/installation.md)'s "Data Dependencies" and "PLLaVA Captioning" sections.


<!-- ### Download the PLLaVA repo

First, make sure you are under the directory of tools/caption/pllava_dir. Then,

```bash
git clone https://github.com/magic-research/PLLaVA.git
cd PLLaVA
git checkout fd9194a


```

### Environment

```bash
conda create -n pllava python=3.10

conda activate pllava

pip install -r requirements.txt # change to your own torch version if neccessary; torch==2.2.2, torchaudio==2.2.2, torchvision==0.17.2 worked for H100 for Tom.

```


### Download weights

```bash
python python_scripts/hf.py # download the weights
``` -->
### Usage

Since PLLaVA is not fashioned as a package, we will use PYTHONPATH to use it.


```bash
cd .. # step back to pllava_dir

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHONPATH='$PYTHONPATH:OPEN_SORA_HOME/tools/caption/pllava_dir/PLLaVA' \
nohup python caption_pllava.py \
  --pretrained_model_name_or_path PLLaVA/MODELS/pllava-13b \
  --use_lora \
  --lora_alpha 4 \
  --num_frames 4 \
  --weight_dir PLLaVA/MODELS/pllava-13b \
  --csv_path meta.csv \
  --pooling_shape 4-12-12 \
  > pllava_caption.out 2>&1 &
```

### PLLaVA vs. LLaVA
In our previous releases, we used [LLaVA](#llava-captioning) for video captioning.
Qualitatively speaking, we observe that PLLaVA has a somewhat higher chance of accurately capture the details in the video than LLaVA. See below for their comparison on a video sample.
<!-- <img src="../../assets/readme/llava_vs_pllava_sample.gif" width="300" height="200" alt="LLaVA vs PLLaVA"> -->

<figure>
    <img src="../../assets/readme/llava_vs_pllava_sample.gif" width="300" height="200" alt="LLaVA vs PLLaVA">
</figure>



| LLaVA | PLLaVA |
|----------|----------|
| The video is a close-up shot of two gold wedding rings. The rings are placed on a (black surface)✅, casting a soft shadow beneath them. The rings are positioned in such a way that (they are facing each other)❌, creating a mirror image effect. The rings are (identical in size and design)✅, suggesting they are a pair. The lighting in the video is soft and diffused, highlighting the gold color of the rings and creating a warm and inviting atmosphere. The overall style of the video is minimalist and elegant, focusing solely on the rings and their reflection.	| The video shows a pair of gold wedding rings on a (reflective surface)✅. The rings are placed one on top of the other, (with the top ring slightly tilted to the left)✅. The rings have a (shiny, metallic finish)✅ and are the main focus of the image. The background is a gradient of dark to light gray, providing a neutral backdrop that highlights the rings. There are no texts or other objects in the image. The style of the video is a simple product display with a focus on the rings, likely intended for promotional or sales purposes. The lighting and shadows suggest a soft, even light source, (possibly a studio light)✅, which creates a reflective surface beneath the rings.|
<!-- |Row2Cell1|Row2Cell2| -->




## LLaVA Captioning

We extract three frames from the video for captioning. With batch inference, we can achieve 10 times speedup. With approximately 720p resolution and 1 frames, the speed is 2~3 videos/s on 8 GPUs. If we resize the smaller side to 336, the speed can be 8 videos/s. In Open-Sora v1.1, to lower the cost, we use the 7B model.

### Installation

Install the required dependancies by following our [installation instructions](../../docs/installation.md)'s "Data Dependencies" and "LLaVA Captioning" sections.

<!-- ### Requirement

```bash
# create conda env
conda create -n llava python=3.10 -y
conda activate llava

# install torch
pip install torch torchvision

# clone llava
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
# CAUTION: This line is to remove torch dependency in pyproject.toml, which is:
# "torch==2.1.2", "torchvision==0.16.2",
# It is better manually remove it in your local pyproject.toml
sed -i '16d' pyproject.toml

# install llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# install flash attention
pip install flash-attn --no-build-isolation
# install colossalai and decord
pip install colossalai decord
``` -->

### Usage

Prepare a csv file for processing. The csv file can be generated by `convert_dataset.py` according to its [documentation](/tools/datasets/README.md). Then, run the following command to generate captions for videos/images with Llava:

```bash
# caption with mistral-7B
torchrun --nproc_per_node 8 --standalone -m tools.caption.caption_llava DATA.csv --dp-size 8 --tp-size 1 --model-path liuhaotian/llava-v1.6-mistral-7b --prompt video

# caption with llava-34B
# NOTE: remember to enable flash attention for this model
torchrun --nproc_per_node 8 --standalone -m tools.caption.caption_llava DATA.csv --dp-size 4 --tp-size 2 --model-path liuhaotian/llava-v1.6-34b --prompt image-3ex --flash-attention

# we run this on 8xH800 GPUs
torchrun --nproc_per_node 8 --standalone -m tools.caption.caption_llava DATA.csv --tp-size 2 --dp-size 4 --bs 16

# at least two 80G GPUs are required
torchrun --nproc_per_node 2 --standalone -m tools.caption.caption_llava DATA.csv --tp-size 2 --dp-size 1 --bs 16

# can also caption images
torchrun --nproc_per_node 2 --standalone -m tools.caption.caption_llava DATA.csv --tp-size 2 --dp-size 1 --bs 16 --prompt image-3ex
```

Please note that you should add the `--flash-attention` flag when running with Llama-based Llava models as it provides speedup but do turn it off for mistral-based ones. Reasons can be found in [this issue](https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453).

After running the script, with `dp-size=N`, you will get `N` parts of csv files. Run the following command to merge them:

```bash
python -m tools.datasets.datautil DATA_caption_part*.csv --output DATA_caption.csv
```

### Resume

Sometimes the process may be interrupted. We can resume the process by running the following command:

```bash
# merge generated results
python -m tools.datasets.datautil DATA_caption_part*.csv --output DATA_caption.csv

# get the remaining videos
python -m tools.datasets.datautil DATA.csv --difference DATA_caption.csv --output DATA_remaining.csv
```

Then use the output csv file to resume the process.


## GPT-4V Captioning

Run the following command to generate captions for videos with GPT-4V:

```bash
# output: DATA_caption.csv
python -m tools.caption.caption_gpt4 DATA.csv --key $OPENAI_API_KEY
```

The cost is approximately $0.01 per video (3 frames per video).

## Camera Motion Detection

<!-- Install additional required packages: `tools/caption/camera_motion/requirements.txt`. -->
Install required packages with `pip install -v .[data]` (See [installation.md](../../docs/installation.md)).
Run the following command to classify camera motion:

```bash
# output: meta_cmotion.csv
python -m tools.caption.camera_motion.detect tools/caption/camera_motion/meta.csv
```

You may additionally specify `threshold` to indicate how "sensitive" the detection should be as below. For example `threshold = 0.2` means that the video is only counted as `tilt_up` when the pixels moved down by `>20%` of video height between the starting and ending frames.
```bash
# output: meta_cmotion.csv
python -m tools.caption.camera_motion.detect tools/caption/camera_motion/meta.csv --threshold 0.2
```

Each video is classified according to 8 categories:
            `pan_right,
            pan_left,
            tilt_up,
            tilt_down,
            zoom_in,
            zoom_out,
            static,
            unclassified`.
Categories of `tilt`, `pan` and `zoom` can overlap with each other.


## Tagging with Llama3

To understand the overall category distribution of our training dataset, we use Llama3 to generate tags based on the video captions.

After obtaining Llama3 usage permission from huggingface/meta, you may generate tags based on the captions using Llama3 like this:

```bash
torchrun --nproc_per_node 8 --standalone -m tools.caption.caption_llama3 meta.csv --key objects --output_prefix meta
```

This will generate tags based on the `text` column of `meta.csv` and put the results to `output_prefix + key.csv`. Currently the prompts for `objects` and `actions` are supported.
