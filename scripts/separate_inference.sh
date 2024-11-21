#!/bin/bash
set_default_params() {
    gpus=${1:-"0,1"}
    num_frames=${2:-"4s"}
    resolution=${3:-"720p"}
    aspect_ratio=${4:-"9:16"}
    aes=${5:-"7"}
    prompt=${6:-"Create a video featuring Will Smith enjoying a plate of spaghetti."}
}

set_default_params "$@"

export CUDA_VISIBLE_DEVICES=$gpus

gpus="${gpus// /}"
IFS=',' read -ra gpu_array <<< "$gpus"
gpu_count=${#gpu_array[@]}

torchrun  --nproc_per_node $gpu_count --master_port=23456 scripts/separate_inference/inference_text_encoder.py configs/opensora-v1-2/inference/sample.py --aes $aes --num-frames "$num_frames" --resolution "$resolution" --aspect-ratio "$aspect_ratio" --prompt "$prompt"
if echo "$prompt" | grep -q "reference_path"; then
    torchrun --nproc_per_node $gpu_count --master_port=23456 scripts/separate_inference/inference_vae_encoder.py configs/opensora-v1-2/inference/sample.py --aes $aes --num-frames "$num_frames" --resolution "$resolution" --aspect-ratio "$aspect_ratio" --prompt "$prompt"
fi
torchrun --nproc_per_node $gpu_count --master_port=23456 scripts/separate_inference/inference_stdit.py configs/opensora-v1-2/inference/sample.py --aes $aes --num-frames "$num_frames" --resolution "$resolution" --aspect-ratio "$aspect_ratio" --prompt "$prompt"
torchrun --nproc_per_node $gpu_count --master_port=23456 scripts/separate_inference/inference_vae_decoder.py configs/opensora-v1-2/inference/sample.py --aes $aes --num-frames "$num_frames" --resolution "$resolution" --aspect-ratio "$aspect_ratio" --prompt "$prompt"
