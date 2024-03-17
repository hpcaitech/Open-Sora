# Video Captioning

Human labeling of videos is expensive and time-consuming. We adopt powerful image captioning models to generate captions for videos. Although GPT-4V achieves a better performance, its 20s/sample speed is too slow for us. With batch inference, we can achieve a speed of 3s/sample with LLaVA, and the quality is comparable. LLaVA is the second best open-source model in [MMMU](https://mmmu-benchmark.github.io/) and accepts any resolution.

![Caption](https://i0.imgs.ovh/2024/03/16/eXdvC.png)

## GPT-4V Captioning

Run the following command to generate captions for videos with GPT-4V:

```bash
python -m tools.caption.caption_gpt4 FOLDER_WITH_VIDEOS output.csv --key $OPENAI_API_KEY
```

The cost is approximately $0.01 per video (3 frames per video). The output is a CSV file with path and caption.

## LLaVA Captioning

First, install LLaVA according to their [official instructions](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install). We use the `liuhaotian/llava-v1.6-34b` model for captioning, which can be download [here](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b). Then, run the following command to generate captions for videos with LLaVA:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m tools.caption.caption_llava samples output.csv
```

The Yi-34B requires 2 80GB GPUs and 3s/sample. The output is a CSV file with path and caption.
