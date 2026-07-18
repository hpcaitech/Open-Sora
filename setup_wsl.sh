#!/bin/bash
# WSL Setup Script for Open-Sora Jelly Blob Video Generation

set -e

echo "=========================================="
echo "Open-Sora WSL Setup"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running in WSL
if ! grep -qi microsoft /proc/version; then
    echo -e "${YELLOW}Warning: This script is designed for WSL. Detected non-WSL environment.${NC}"
fi

echo -e "${BLUE}Step 1: Update system packages${NC}"
sudo apt-get update
sudo apt-get upgrade -y

echo -e "${BLUE}Step 2: Install CUDA Toolkit${NC}"
wget -O /tmp/cuda_12.1_installer.run https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh /tmp/cuda_12.1_installer.run --silent --driver

echo -e "${BLUE}Step 3: Install cuDNN${NC}"
# Note: You may need to download cuDNN manually from NVIDIA website
# Extract to /usr/local/cuda/

echo -e "${BLUE}Step 4: Install Python and build tools${NC}"
sudo apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    build-essential \
    python3.10-dev \
    ffmpeg

echo -e "${BLUE}Step 5: Create virtual environment${NC}"
python3.10 -m venv ~/opensora-env
source ~/opensora-env/bin/activate

echo -e "${BLUE}Step 6: Upgrade pip${NC}"
pip install --upgrade pip setuptools wheel

echo -e "${BLUE}Step 7: Install PyTorch with CUDA support${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo -e "${BLUE}Step 8: Navigate to Open-Sora directory${NC}"
cd /mnt/c/Users/amink/.copilot/repos/copilot-worktrees/Open-Sora/salihcamal2-art-ideal-telegram

echo -e "${BLUE}Step 9: Install Open-Sora and dependencies${NC}"
pip install -e . --no-build-isolation

echo -e "${GREEN}=========================================="
echo "Setup complete!"
echo "==========================================${NC}"

echo -e "${BLUE}To generate the video, run:${NC}"
echo ""
echo "  source ~/opensora-env/bin/activate"
echo "  cd /mnt/c/Users/amink/.copilot/repos/copilot-worktrees/Open-Sora/salihcamal2-art-ideal-telegram"
echo "  python scripts/inference.py \\"
echo "    --config configs/opensora/inference/16x256x256.py \\"
echo "    --prompt-path jelly_blob_prompt.txt \\"
echo "    --save-dir ./outputs/jelly_blob_video"
echo ""
