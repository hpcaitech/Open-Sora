# Open-Sora 1.1 Report

- [Model Architecture Modification](#model-architecture-modification)
- [Support for Multi-time/resolution/aspect ratio/fps Training](#support-for-multi-timeresolutionaspect-ratiofps-training)
- [Masked DiT as Image/Video-to-Video Model](#masked-dit-as-imagevideo-to-video-model)
- [Data Collection \& Pipeline](#data-collection--pipeline)
- [Training Details](#training-details)
- [Limitation and Future Work](#limitation-and-future-work)

In Open-Sora 1.1 release, we train a 700M models on 10M data (Open-Sora 1.0 trained on 400K data) with a better STDiT architecture. We implement the following features mentioned in [sora's report](https://openai.com/research/video-generation-models-as-world-simulators):

- Variable durations, resolutions, aspect ratios (Sampling flexibility, Improved framing and composition)
- Prompting with images and videos (Animating images, Extending generated videos, Video-to-video editing, Connecting videos)
- Image generation capabilities

To achieve this goal, we use multi-task learning in the pretraining stage. For diffusion models, training with different sampled timestep is already a multi-task learning. We further extend this idea to multi-resolution, aspect ratio, frame length, fps, and different mask strategies for image and video conditioned generation. We train the model on **0s~15s, 144p to 720p, various aspect ratios** videos. Although the quality of time consistency is not that high due to limit training FLOPs, we can still see the potential of the model.

## Model Architecture Modification

We made the following modifications to the original ST-DiT for better training stability and performance (ST-DiT-2):

- **[Rope embedding](https://arxiv.org/abs/2104.09864) for temporal attention**: Following LLM's best practice, we change the sinusoidal positional encoding to rope embedding for temporal attention since it is also a sequence prediction task.
- **AdaIN and Layernorm for temporal attention**: we wrap the temporal attention with AdaIN and layernorm as the spatial attention to stabilize the training.
- **[QK-normalization](https://arxiv.org/abs/2302.05442) with [RMSNorm](https://arxiv.org/abs/1910.07467)**: Following [SD3](https://arxiv.org/pdf/2403.03206.pdf), we apply QK-normalization to the all attention for better training stability in half-precision.
- **Dynamic input size support and video infomation condition**: To support multi-resolution, aspect ratio, and fps training, we make ST-DiT-2 to accept any input size, and automatically scale positional embeddings. Extending [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha)'s idea, we conditioned on video's height, width, aspect ratio, frame length, and fps.
- **Extending T5 tokens from 120 to 200**: our caption is usually less than 200 tokens, and we find the model can handle longer text well.

## Support for Multi-time/resolution/aspect ratio/fps Training

As mentioned in the [sora's report](https://openai.com/research/video-generation-models-as-world-simulators), training with original video's resolution, aspect ratio, and length increase sampling flexibility and improve framing and composition. We found three ways to achieve this goal:

- [NaViT](https://arxiv.org/abs/2307.06304): support dynamic size within the same batch by masking, with little efficiency loss. However, the system is a bit complex to implement, and may not benefit from optimized kernels such as flash attention.
- Padding ([FiT](https://arxiv.org/abs/2402.12376), [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan)): support dynamic size within the same batch by padding. However, padding different resolutions to the same size is not efficient.
- Bucket ([SDXL](https://arxiv.org/abs/2307.01952), [PixArt](https://arxiv.org/abs/2310.00426)): support dynamic size in different batches by bucketing, but the size must be the same within the same batch, and only a fixed number of size can be applied. With the same size in a batch, we do not need to implement complex masking or padding.

For the simplicity of implementation, we choose the bucket method. We pre-define some fixed resolution, and allocate different samples to different bucket. The concern for bucketing is listed below. But we can see that the concern is not a big issue in our case.

<details>
<summary>View the concerns</summary>

- The bucket size is limited to a fixed number: First, in real-world applications, only a few aspect ratios (9:16, 3:4) and resolutions (240p, 1080p) are commonly used. Second, we find trained models can generalize well to unseen resolutions.
- The size in each batch is the same, breaks the i.i.d. assumption: Since we are using multiple GPUs, the local batches on different GPUs have different sizes. We did not see a significant performance drop due to this issue.
- The may not be enough samples to fill each bucket and the distribution may be biased: First, our dataset is large enough to fill each bucket when local batch size is not too large. Second, we should analyze the data's distribution on sizes and define the bucket size accordingly. Third, an unbalanced distribution did not affect the training process significantly.
- Different resolutions and frame lengths may have different processing speed: Different from PixArt, which only deals with aspect ratios of similar resolutions (similar token numbers), we need to consider the processing speed of different resolutions and frame lengths. We can use the `bucket_config` to define the batch size for each bucket to ensure the processing speed is similar.

</details>

![bucket](/assets/readme/report_bucket.png)

As shown in the figure, a bucket is a triplet of `(resolution, num_frame, aspect_ratio)`. We provide pre-defined aspect ratios for different resolution that covers most of the common video aspect ratios. Before each epoch, we shuffle the dataset and allocate the samples to different buckets as shown in the figure. We put a sample into a bucket with largest resolution and frame length that is smaller than the video's.

Considering our computational resource is limited, we further introduce two attributes `keep_prob` and `batch_size` for each `(resolution, num_frame)` to reduce the computational cost and enable multi-stage training. Specifically, a high-resolution video will be downsampled to a lower resolution with probability `1-keep_prob` and the batch size for each bucket is `batch_size`. In this way, we can control the number of samples in different buckets and balance the GPU load by search a good batch size for each bucket.

A detailed explanation of the bucket usage in training is available in [docs/config.md](/docs/config.md#training-bucket-configs).

## Masked DiT as Image/Video-to-Video Model

Transformers can be easily extended to support image-to-image and video-to-video tasks. We propose a mask strategy to support image and video conditioning. The mask strategy is shown in the figure below.

![mask strategy](/assets/readme/report_mask.png)

Typically, we unmask the frames to be conditioned on for image/video-to-video condition. During the ST-DiT forward, unmasked frames will have timestep 0, while others remain the same (t). We find directly apply the strategy to trained model yield poor results as the diffusion model did not learn to handle different timesteps in one sample during training.

Inspired by [UL2](https://arxiv.org/abs/2205.05131), we introduce random mask strategy during training. Specifically, we randomly unmask the frames during training, including unmask the first frame, the first k frames, the last frame, the last k frames, the first and last k frames, random frames, etc. Based on Open-Sora 1.0, with 50% probability of applying masking, we see the model can learn to handle image conditioning (while 30% yields worse ability) for 10k steps, with a little text-to-video performance drop. Thus, for Open-Sora 1.1, we pretrain the model from scratch with masking strategy.

An illustration of masking strategy config to use in inference is given as follow. A five number tuple provides great flexibility in defining the mask strategy. By conditioning on generated frames, we can autogressively generate infinite frames (although error propagates).

![mask strategy config](/assets/readme/report_mask_config.png)

A detailed explanation of the mask strategy usage is available in [docs/config.md](/docs/config.md#advanced-inference-config).

## Data Collection & Pipeline

As we found in Open-Sora 1.0, the data number and quality are crucial for training a good model, we work hard on scaling the dataset. First, we create an automatic pipeline following [SVD](https://arxiv.org/abs/2311.15127), inlcuding scene cutting, captioning, various scoring and filtering, and dataset management scripts and conventions. More infomation can be found in [docs/data_processing.md](/docs/data_processing.md).

![pipeline](/assets/readme/report_data_pipeline.png)

We plan to use [panda-70M](https://snap-research.github.io/Panda-70M/) and other data to traing the model, which is approximately 30M+ data. However, we find disk IO a botteleneck for training and data processing at the same time. Thus, we can only prepare a 10M dataset and did not go through all processing pipeline that we built. Finally, we use a dataset with 9.7M videos + 2.6M images for pre-training, and 560k videos + 1.6M images for fine-tuning. The pretraining dataset statistics are shown below. More information about the dataset can be found in [docs/datasets.md](/docs/datasets.md).

Image text tokens (by T5 tokenizer):

![image text tokens](/assets/readme/report_image_textlen.png)

Video text tokens (by T5 tokenizer). We directly use panda's short caption for training, and caption other datasets by ourselves. The generated caption is usually less than 200 tokens.

![video text tokens](/assets/readme/report_video_textlen.png)

Video duration:

![video duration](/assets/readme/report_video_duration.png)

## Training Details

With limited computational resources, we have to carefully monitor the training process, and change the training strategy if we speculate the model is not learning well since there is no computation for ablation study. Thus, Open-Sora 1.1's training includes multiple changes, and as a result, ema is not applied.

1. First, we fine-tune **6k** steps with images of different resolution from `Pixart-alpha-1024` checkpoints. We find the model easily adapts to generate images with different resolutions. We use [SpeeDiT](https://github.com/1zeryu/SpeeDiT) (iddpm-speed) to accelerate the diffusion training.
2. **[Stage 1]** Then, we pretrain the model with gradient-checkpointing for **24k** steps, which takes **4 days** on 64 H800 GPUs. Although the number of samples seen by the model is the same, we find the model learns slowly compared to a smaller batch size. We speculate that at an early stage, the number of steps is more important for training. The most videos are in **240p** resolution, and the config is similar to [stage2.py](/configs/opensora-v1-1/train/stage2.py). The video looking is good, but the model does not know much about the temporal knowledge. We use mask ratio of 10%.
3. **[Stage 1]** To increase the number of steps, we switch to a smaller batch size without gradient-checkpointing. We also add fps conditioning at this point. We trained **40k** steps for **2 days**. The most videos are in **144p** resolution, and the config file is [stage1.py](/configs/opensora-v1-1/train/stage1.py). We use a lower resolution as we find in Open-Sora 1.0 that the model can learn temporal knowledge with relatively low resolution.
4. **[Stage 1]** We find the model cannot learn well for long videos, and find a noised generation result as speculated to be half-precision problem found in Open-Sora 1.0 training. Thus, we adopt the QK-normalization to stabilize the training. Similar to SD3, we find the model quickly adapt to the QK-normalization. We also switch iddpm-speed to iddpm, and increase the mask ratio to 25% as we find image-condition not learning well. We trained for **17k** steps for **14 hours**. The most videos are in **144p** resolution, and the config file is [stage1.py](/configs/opensora-v1-1/train/stage1.py). The stage 1 training lasts for approximately one week, with total step **81k**.
5. **[Stage 2]** We switch to a higher resolution, where most videos are in **240p and 480p** resolution ([stage2.py](/configs/opensora-v1-1/train/stage2.py)). We trained **22k** steps for **one day** on all pre-training data.
6. **[Stage 3]** We switch to a higher resolution, where most videos are in **480p and 720p** resolution ([stage3.py](/configs/opensora-v1-1/train/stage3.py)). We trained **4k** with **one day** on high-quality data. We find loading previous stage's optimizer state can help the model learn faster.

To summarize, the training of Open-Sora 1.1 requires approximately **9 days** on 64 H800 GPUs.

## Limitation and Future Work

As we get one step closer to the replication of Sora, we find many limitations for the current model, and these limitations point to the future work.

- **Generation Failure**: we fine many cases (especially when the total token number is large or the content is complex),  our model fails to generate the scene. There may be a collapse in the temporal attention and we have identified a potential bug in our code. We are working hard to fix it. Besides, we will increase our model size and training data to improve the generation quality in the next version.
- **Noisy generation and influency**: we find the generated model is sometimes noisy and not fluent, especially for long videos. We think the problem is due to not using a temporal VAE. As [Pixart-Sigma](https://arxiv.org/abs/2403.04692) finds that adapting to a new VAE is simple, we plan to develop a temporal VAE for the model in the next version.
- **Lack of time consistency**: we find the model cannot generate videos with high time consistency. We think the problem is due to the lack of training FLOPs. We plan to collect more data and continue training the model to improve the time consistency.
- **Bad human generation**: We find the model cannot generate high-quality human videos. We think the problem is due to the lack of human data. We plan to collect more human data and continue training the model to improve the human generation.
- **Low aesthetic score**: we find the model's aesthetic score is not high. The problem is due to the lack of aesthetic score filtering, which is not conducted due to IO bottleneck. We plan to filter the data by aesthetic score and finetuning the model to improve the aesthetic score.
- **Worse quality for longer video generation**: we find with a same prompt, the longer video has worse quality. This means the image quality is not equally adapted to different lengths of sequences.

> - **Algorithm & Acceleration**: Zangwei Zheng, Xiangyu Peng, Shenggui Li, Hongxing Liu, Yukun Zhou, Tianyi Li
> - **Data Collection & Pipeline**: Xiangyu Peng, Zangwei Zheng, Chenhui Shen, Tom Young, Junjie Wang, Chenfeng Yu
