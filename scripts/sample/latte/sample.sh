#!/usr/bin/env bash

# get args
GPUS=${1:-8}

# get root dir
FOLDER_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR=$FOLDER_DIR/../../..

# go to root dir
cd $ROOT_DIR


export PYTHONPATH=$FOLDER_DIR:$PYTHONPATH
python $FOLDER_DIR/sample_t2v.py \
    --checkpoint /home/lishenggui/projects/sora/hf-weights/models--maxin-cn--Latte/snapshots/8f0591220fa329f9d917086810b3c0f6544a87c7/t2v.pt \
    --model_path /home/lishenggui/projects/sora/hf-weights/models--maxin-cn--Latte/snapshots/8f0591220fa329f9d917086810b3c0f6544a87c7/t2v_required_models/ \
    --text_prompt "A dog in astronaut suit and sunglasses floating in space" \
    --output_path $ROOT_DIR/outputs/latte
