#!/bin/bash
# Docker setup and run script for Open-Sora video generation

set -e

echo "=========================================="
echo "Open-Sora Docker Setup & Video Generation"
echo "=========================================="

REPO_PATH="/mnt/c/Users/amink/.copilot/repos/copilot-worktrees/Open-Sora/salihcamal2-art-ideal-telegram"
DOCKER_IMAGE="opensora-jelly-blob"
DOCKER_TAG="latest"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
fi

echo -e "${BLUE}Building Docker image...${NC}"
docker build -f Dockerfile.opensora -t $DOCKER_IMAGE:$DOCKER_TAG .

echo -e "${BLUE}Running Docker container...${NC}"
docker run --rm \
    --gpus all \
    -v "$REPO_PATH:/workspace" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -it $DOCKER_IMAGE:$DOCKER_TAG \
    bash -c "
        cd /workspace && \
        python scripts/inference.py \
            --config configs/opensora/inference/16x256x256.py \
            --prompt-path jelly_blob_prompt.txt \
            --save-dir ./outputs/jelly_blob_video && \
        echo 'Video generation complete!'
    "

echo -e "${GREEN}Video saved to: $REPO_PATH/outputs/jelly_blob_video/${NC}"
