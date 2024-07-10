#!/bin/bash
set_default_params() {
    num_frames=${1:-"4s"}
    resolution=${2:-"720p"}
    aspect_ratio=${3:-"9:16"}
    prompt=${4:-"Create a video featuring Will Smith enjoying a plate of spaghetti."}
}

set_default_params "$@"

CUDA_VISIBLE_DEVICES=0,1 torchrun  --nproc_per_node 2 --master_port=23456 scripts/separate_inference/inference_text_encoder.py configs/opensora-v1-2/inference/sample.py --num-frames "$num_frames" --resolution "$resolution" --aspect-ratio "$aspect_ratio" --prompt "$prompt"
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=23456 scripts/separate_inference/inference_stdit.py configs/opensora-v1-2/inference/sample.py --num-frames "$num_frames" --resolution "$resolution" --aspect-ratio "$aspect_ratio" --prompt "$prompt"
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=23456 scripts/separate_inference/inference_vae_decoder.py configs/opensora-v1-2/inference/sample.py --num-frames "$num_frames" --resolution "$resolution" --aspect-ratio "$aspect_ratio" --prompt "$prompt"
