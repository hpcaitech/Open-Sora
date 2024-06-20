# Repo Structure

```plaintext
Open-Sora
├── README.md
├── assets
│   ├── images                     -> images used for image-conditioned generation
│   ├── demo                       -> images used for demo
│   ├── texts                      -> prompts used for text-conditioned generation
│   └── readme                     -> images used in README
├── configs                        -> Configs for training & inference
├── docker                         -> dockerfile for Open-Sora
├── docs
│   ├── acceleration.md            -> Report on acceleration & speed benchmark
│   ├── commands.md                -> Commands for training & inference
│   ├── datasets.md                -> Datasets used in this project
|   ├── data_processing.md         -> Data pipeline documents
|   ├── installation.md            -> Data pipeline documents
│   ├── structure.md               -> This file
│   ├── config.md                  -> Configs for training and inference
│   ├── report_01.md               -> Report for Open-Sora 1.0
│   ├── report_02.md               -> Report for Open-Sora 1.1
│   ├── report_03.md               -> Report for Open-Sora 1.2
│   ├── vae.md                     -> our VAE report
│   └── zh_CN                      -> Chinese version of the above
├── eval                           -> Evaluation scripts
│   ├── README.md                  -> Evaluation documentation
|   ├── human_eval                 -> for human eval
|   ├── launch.sh                  -> script for launching 8 cards sampling
|   ├── loss                       -> eval loss
|   ├── sample.sh                  -> script for quickly launching inference on predefined prompts
|   ├── vae                        -> for vae eval
|   ├── vbench                     -> for VBench evaluation
│   └── vbench_i2v                 -> for VBench i2v evaluation
├── gradio                         -> Gradio demo related code
├── notebooks                      -> Jupyter notebooks for generating commands to run
├── scripts
│   ├── train.py                   -> diffusion training script
│   ├── train_vae.py               -> vae training script
│   ├── inference.py               -> diffusion inference script
│   ├── inference_vae.py           -> vae inference script
│   └── misc                       -> misc scripts, including batch size search
├── opensora
│   ├── __init__.py
│   ├── registry.py                -> Registry helper
│   ├── acceleration               -> Acceleration related code
│   ├── datasets                    -> Dataset related code
│   ├── models
│   │   ├── dit                    -> DiT
│   │   ├── layers                 -> Common layers
│   │   ├── vae                    -> VAE as image encoder
│   │   ├── text_encoder           -> Text encoder
│   │   │   ├── classes.py         -> Class id encoder (inference only)
│   │   │   ├── clip.py            -> CLIP encoder
│   │   │   └── t5.py              -> T5 encoder
│   │   ├── dit
│   │   ├── latte
│   │   ├── pixart
│   │   └── stdit                  -> Our STDiT related code
│   ├── schedulers                 -> Diffusion schedulers
│   │   ├── iddpm                  -> IDDPM for training and inference
│   │   └── dpms                   -> DPM-Solver for fast inference
│   └── utils
├── tests                          -> Tests for the project
└── tools                          -> Tools for data processing and more
```

## Configs

Our config files follows [MMEgine](https://github.com/open-mmlab/mmengine). MMEngine will reads the config file (a `.py` file) and parse it into a dictionary-like object.

```plaintext
Open-Sora
└── configs                        -> Configs for training & inference
    ├── opensora-v1-1              -> STDiT2 related configs
    │   ├── inference
    │   │   ├── sample.py          -> Sample videos and images
    │   │   └── sample-ref.py      -> Sample videos with image/video condition
    │   └── train
    │       ├── stage1.py          -> Stage 1 training config
    │       ├── stage2.py          -> Stage 2 training config
    │       ├── stage3.py          -> Stage 3 training config
    │       ├── image.py           -> Illustration of image training config
    │       ├── video.py           -> Illustration of video training config
    │       └── benchmark.py       -> For batch size searching
    ├── opensora                   -> STDiT related configs
    │   ├── inference
    │   │   ├── 16x256x256.py      -> Sample videos 16 frames 256x256
    │   │   ├── 16x512x512.py      -> Sample videos 16 frames 512x512
    │   │   └── 64x512x512.py      -> Sample videos 64 frames 512x512
    │   └── train
    │       ├── 16x256x256.py      -> Train on videos 16 frames 256x256
    │       ├── 16x256x256.py      -> Train on videos 16 frames 256x256
    │       └── 64x512x512.py      -> Train on videos 64 frames 512x512
    ├── dit                        -> DiT related configs
    │   ├── inference
    │   │   ├── 1x256x256-class.py -> Sample images with ckpts from DiT
    │   │   ├── 1x256x256.py       -> Sample images with clip condition
    │   │   └── 16x256x256.py      -> Sample videos
    │   └── train
    │       ├── 1x256x256.py       -> Train on images with clip condition
    │       └── 16x256x256.py      -> Train on videos
    ├── latte                      -> Latte related configs
    └── pixart                     -> PixArt related configs
```

## Tools

```plaintext
Open-Sora
└── tools
    ├── datasets                   -> dataset management related code
    ├── scene_cut                  -> scene cut related code
    ├── caption                    -> caption related code
    ├── scoring                    -> scoring related code
    │   ├── aesthetic              -> aesthetic scoring related code
    │   ├── matching               -> matching scoring related code
    │   ├── ocr                    -> ocr scoring related code
    │   └── optical_flow           -> optical flow scoring related code
    └── frame_interpolation        -> frame interpolation related code
