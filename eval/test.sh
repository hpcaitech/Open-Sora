#!/bin/bash

# set -x
set -e

CKPT=$1
NUM_FRAMES=$2
MODEL_NAME=$3

let DOUBLE_FRAMES=$2*2
let TRIPLE_FRAMES=$2*3

echo $DOUBLE_FRAMES
echo $TRIPLE_FRAMES
