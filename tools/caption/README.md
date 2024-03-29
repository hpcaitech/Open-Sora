# Video Captioning

Human labeling of videos is expensive and time-consuming. We adopt powerful image captioning models to generate captions for videos. Although GPT-4V achieves a better performance, its 20s/sample speed is too slow for us. LLaVA is the second best open-source model in [MMMU](https://mmmu-benchmark.github.io/) and accepts any resolution. We find the quality of 34B model is comparable.

![Caption](https://i0.imgs.ovh/2024/03/16/eXdvC.png)

We extract three frames from the video for captioning. With batch inference, we can achieve 10 times speedup, with 2.4 videos/s on 8 GPUs.

## GPT-4V Captioning

Run the following command to generate captions for videos with GPT-4V:

```bash
python -m tools.caption.caption_gpt4 FOLDER_WITH_VIDEOS output.csv --key $OPENAI_API_KEY
```

The cost is approximately $0.01 per video (3 frames per video). The output is a CSV file with path and caption.

## LLaVA Captioning

### Requirement

```bash
# create conda env
conda create -n llava python=3.10 -y
conda activate llava

# install llava
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# 如果你在英博机器上，需要重新安装torch
pip uninstall torch torchvision torchaudio
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# install flash attention
pip install flash-attn --no-build-isolation
```

First, install LLaVA according to their [official instructions](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install). We use the `liuhaotian/llava-v1.6-34b` model for captioning, which can be download [here](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b). 

### Usage

Then, run the following command to generate captions for videos with LLaVA:

```bash
# we run this on 8xH800 GPUs
torchrun --nproc_per_node 8 --standalone -m tools.caption.caption_llava samples output.csv --tp-size 2 --dp-size 4 --bs 16
```
