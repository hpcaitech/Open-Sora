#!/bin/bash

# Automated setup script for Open-Sora in a single folder

# Create the main folder
mkdir -p ~/Open-Sora-All

# Navigate to the folder
cd ~/Open-Sora-All

# Clone the Open-Sora repository directly into the current folder
git clone https://github.com/hpcaitech/Open-Sora .

# Create a virtual environment inside the folder
python3 -m venv sora-env

# Activate the virtual environment
source sora-env/bin/activate

# Edit requirements.txt for CPU-only compatibility
# Change torch to torch==2.4.0
sed -i '' 's/^torch.*/torch==2.4.0/' requirements.txt

# Change torchvision to torchvision==0.19.0
sed -i '' 's/^torchvision.*/torchvision==0.19.0/' requirements.txt

# Comment out GPU-only packages: triton and liger-kernel
sed -i '' '/^triton/s/^/#/' requirements.txt
sed -i '' '/^liger-kernel/s/^/#/' requirements.txt

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create models folder
mkdir models

# Note: Download model files as per Open-Sora README instructions
echo "Setup complete. Please download required model files into ~/Open-Sora-All/models/ as per the Open-Sora README instructions."
echo "To run video generation, navigate to ~/Open-Sora-All, activate the venv with 'source sora-env/bin/activate', and run your commands."
