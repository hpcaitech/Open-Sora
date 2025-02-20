# Open-Sora 1.3 Report

- [Video compression network](#video-compression-network)
- [Upgraded STDiT with shifted-window attention](#upgraded-stdit-with-shifted-window-attention)
- [Easy and effective model conditioning](#easy-and-effective-model-conditioning)
- [Evaluation](#evaluation)

In Open-Sora 1.3 release, we train a 1.1B models on >60M data (\~85k hours), with training cost 35k H100 GPU hours, supporting 0s\~113 frames, 360p & 720p, various aspect ratios video generation. Our configurations is listed below. Following our 1.2 version, Open-Sora 1.3 can also do image-to-video generation and video extension.

|      | image | 49 frames  | 65 frames  | 81 frames  | 97 frames | 113 frames |
| ---- | ----- | ---------- | ---------- | ---------- | --------- | ---------- |
| 360p | ✅     | ✅         | ✅         | ✅         | ✅         |✅          |
| 720p | ✅     | ✅         | ✅         | ✅         | ✅         |✅          |

Here ✅ means that the data is seen during training.

Besides features introduced in Open-Sora 1.2, Open-Sora 1.3 highlights:

- Video compression network
- Upgraded STDiT with shifted-window attention
- More data and better multi-stage training
- Easy and effective model conditioning
- Better evaluation metrics

All implementations (both training and inference) of the above improvements are available in the Open-Sora 1.3 release. The following sections will introduce the details of the improvements. We also refine our codebase and documentation to make it easier to use and develop, and add a LLM refiner to [refine input prompts](/README.md#gpt-4o-prompt-refinement) and support more languages.

## Video compression network

In Open-Sora 1.2, the video compression architecture employed a modular approach, where spatial and temporal dimensions were handled separately. The spatial VAE, based on Stability AI's SDXL VAE, compressed individual frames along the spatial dimensions. The temporal VAE then processed the latent representations from the spatial VAE to handle temporal compression. This two-stage design allowed effective spatial and temporal compression but introduced limitations. These included inefficiencies in handling long videos due to fixed-length input frames, a lack of seamless integration between spatial and temporal features, and higher memory requirements during both training and inference.

Open-Sora 1.3 introduces a unified approach to video compression. By combining spatial and temporal processing into a single framework and leveraging advanced features like tiled 3D convolutions and dynamic frame support, Open-Sora 1.3 achieves improved better efficiency, scalability, and reconstruction quality. Here are the key improvements in Open-Sora 1.3 VAE:

**1. Unified Spatial-Temporal Processing:** Instead of using separate VAEs for spatial and temporal compression, Open-Sora 1.3 adopts a single encoder-decoder structure that simultaneously handles both dimensions. This approach eliminates the need for intermediate representations and redundant data transfers between spatial and temporal modules.

**2. Tiled 3D Convolutions:** Open-Sora 1.3 incorporates tiled 3D convolution support for the temporal dimension. By breaking down videos into smaller temporal tiles, this feature enables efficient encoding and decoding of longer video sequences without increasing memory overhead. This improvement addresses the limitations of Open-Sora 1.2 in handling large frame counts and ensures higher flexibility in temporal compression.

**3. Dynamic Micro-Batch and Micro-Frame Processing:** Open-Sora 1.3 introduces a new micro-batch and micro-frame processing mechanism. This allows for: (1) Adaptive temporal overlap: Overlapping frames during temporal encoding and decoding help reduce discontinuities at tile boundaries. (2) Dynamic frame size support: Instead of being restricted to fixed-length sequences (e.g., 17 frames in Open-Sora 1.2), Open-Sora 1.3 supports dynamic sequence lengths, making it robust for varied video lengths.

**4. Unified Normalization Mechanism:** The normalization process in Open-Sora 1.3 has been refined with tunable scaling (scale) and shifting (shift) parameters that ensure consistent latent space distributions across diverse datasets. Unlike Open-Sora 1.2, where normalization was specific to fixed datasets, this version introduces more generalized parameters and support for frame-specific normalization strategies.


#### Summary of Improvements

| Feature                | Open-Sora 1.2                          | Open-Sora 1.3                          |
|------------------------|-----------------------------------------|-----------------------------------------|
| **Architecture**       | Separate spatial and temporal VAEs      | Unified spatial-temporal VAE            |
| **Tiled Processing**   | Not supported                          | Supported (Tiled 3D Convolutions)       |
| **Frame Length Support**| Fixed (17 frames)                      | Dynamic frame support with overlap      |
| **Normalization**      | Fixed parameters                       | Tunable scaling and shifting            |


## Upgraded STDiT with shifted-window attention

Following the success of OpenSora 1.2, version 1.3 introduces several architectural improvements and new capabilities to enhance video generation quality and flexibility. This section outlines the key improvements and differences between these two versions.

Latest diffusion models like Stable Diffusion 3 adopt the [rectified flow](https://github.com/gnobitab/RectifiedFlow) instead of DDPM for better performance. While SD3's rectified flow training code is not open-sourced, OpenSora provides the training code following SD3's paper. OpenSora 1.2 introduced several key strategies from SD3:

1. Basic rectified flow training, which enables continuous-time diffusion
2. Logit-norm sampling for training acceleration (following SD3 paper Section 3.1), preferentially sampling timesteps at middle noise levels
3. Resolution and video length aware timestep sampling (following SD3 paper Section 5.3.2), using more noise for larger resolutions and longer videos

For OpenSora 1.3, we further enhance the model with significant improvements in architecture, capabilities, and performance:

#### 1. Shift-Window Attention Mechanism
- Introduced kernel-based local attention with configurable kernel_size for efficient computation
- Implemented shift-window partitioning strategy similar to Swin Transformer
- Added padding mask handling for window boundaries with extra_pad_on_dims support
- Extended position encoding with 3D relative positions within local windows (temporal, height, width)
#### 2. Enhanced Position Encoding
- Improved RoPE implementation with reduced rotation_dim (1/3 of original) for 3D scenarios
- Added separate rotary embeddings for temporal, height, and width dimensions
- Implemented resolution-adaptive scaling for position encodings
- Optional spatial RoPE for better spatial relationship modeling
#### 3. Flexible Generation
- Added I2V and V2V capabilities with dedicated conditioning mechanisms
- Introduced conditional embedding modules (x_embedder_cond and x_embedder_cond_mask)
- Zero-initialized condition embeddings for stable training
- Flexible temporal modeling with skip_temporal option
#### 4. Performance Optimization
- Refined Flash Attention triggering conditions (N > 128) for better efficiency
- Added support for torch.scaled_dot_product_attention (SDPA) as an alternative backend
- Optimized memory usage through improved padding and window partitioning
- Enhanced sequence parallelism with adaptive height padding

The adaptation process from [PixArt-Σ 2K](https://github.com/PixArt-alpha/PixArt-sigma) remains similar but with additional steps:
1-7. [Same as v1.2: multi-resolution training, QK-norm, rectified flow, logit-norm sampling, smaller AdamW epsilon, new VAE, and basic temporal attention]
#### 8. Enhanced temporal blocks
   - Added kernel-based local attention with shift-window support
   - Implemented 3D relative position encoding with resolution-adaptive scaling
   - Zero-initialized projection layers with improved initialization strategy

Compared to v1.2 which focused on basic video generation, v1.3 brings substantial improvements in three key areas: **1. Quality**: Enhanced spatial-temporal modeling through shift-window attention and 3D position encoding. **2. Flexibility**: Support for I2V/V2V tasks and configurable temporal modeling. **3. Efficiency**: Optimized attention computation and memory usage

These improvements maintain backward compatibility with v1.2's core features while extending the model's capabilities for real-world applications. The model retains its ability to generate high-quality images and videos using rectified flow, while gaining new strengths in conditional generation and long sequence modeling.

## Easy and effective model conditioning

We calculate the aesthetic score and motion score for each video clip, and filter out those clips with low scores, which leads to a dataset with better video quality. Additionally, we append the scores to the captions and use them as conditioning. Specifically, we convert numerical scores into descriptive language based on predefined ranges. The aesthetic score transformation function converts numerical aesthetic scores into descriptive labels based on predefined ranges: scores below 4 are labeled "terrible," progressing through "very poor," "poor," "fair," "good," and "very good," with scores of 6.5 or higher labeled as "excellent." Similarly, the motion score transformation function maps motion intensity scores to descriptors: scores below 0.5 are labeled "very low," progressing through "low," "fair," "high," and "very high," with scores of 20 or more labeled as "extremely high." We find this method can make model aware of the scores and follows the scores to generate videos with better quality.

For example, a video with aesthetic score 5.5, motion score 10, and a detected camera motion pan left, the caption will be:

```plaintext
[Original Caption] The aesthetic score is good, the motion strength is high, camera motion: pan left.
```

During inference, we can also use the scores to condition the model. For camera motion, we only label 13k clips with high confidence, and the camera motion detection module is released in our tools.

## Evaluation

Previously, we monitor the training process only by human evaluation, as DDPM traning loss is not well correlated with the quality of generated videos. However, for rectified flow, we find the training loss is well correlated with the quality of generated videos as stated in SD3. Thus, we keep track of rectified flow evaluation loss on 100 images and 1k videos.

We sampled 1k videos from pixabay as validation dataset. We calculate the evaluation loss for image and different lengths of videos (49 frames, 65 frames, 81 frames, 97 frames, 113 frames) for different resolution (360p, 720p). For each setting, we equidistantly sample 10 timesteps. Then all the losses are averaged.

In addition, we also keep track of [VBench](https://vchitect.github.io/VBench-project/) scores during training. VBench is an automatic video evaluation benchmark for short video generation. We calcuate the vbench score with 360p 49-frame videos. The two metrics verify that our model continues to improve during training.

All the evaluation code is released in `eval` folder. Check the [README](/eval/README.md) for more details.
