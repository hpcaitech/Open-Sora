# 10× inference speedup with high-compression autoencoder


The high computational cost of training video generation models arises from the
large number of tokens and the dominance of attention computation. To further reduce training expenses,
we explore training video generation models with high-compression autoencoders (Video DC-AEs). As shown in the comparason below, by switching to the Video DC-AE with a much higher downsample ratio (4 x 32 x 32), we can afford to further reduce the patch size to 1 and still achieve __5.2× speedup in training throughput__ and __10x speedup during inference__:

![opensorav2_speed](https://github.com/hpcaitech/Open-Sora-Demo/blob/main/readme/hcae_opensorav2_speed.png)


Nevertheless, despite the advantanges in drastically lower computation costs, other challenges remain. For instance, larger channels low down convergance. Our generation model adapted with a 128-channel Video DC-AE for 25K iterations achieves a loss level of 0.5, as compared to 0.1 from the initialization model. While the fast video generation model underperforms the original, it still captures spatial-temporal
relationships. We release this model to the research community for further exploration.

Checkout more details in our [report](https://arxiv.org/abs/2503.09642v1).

## Model Download

Download from 🤗 [Huggingface](https://huggingface.co/hpcai-tech/Open-Sora-v2-Video-DC-AE):

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download hpcai-tech/Open-Sora-v2-Video-DC-AE --local-dir ./ckpts
```

## Inference

To inference on our fast video generation model:

```bash
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py configs/diffusion/inference/high_compression.py --prompt "The story of a robot's life in a cyberpunk setting." 
```

## Training
Follow this [guide](./train.md#prepare-dataset) to parepare the __DATASET__ for training.
Then, you may train your own fast generation model with the following command:
```bash
torchrun --nproc_per_node 8 scripts/diffusion/train.py configs/diffusion/train/high_compression.py --dataset.data-path datasets/pexels_45k_necessary.csv
```
