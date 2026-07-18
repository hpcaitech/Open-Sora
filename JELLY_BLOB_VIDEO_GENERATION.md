# Jelly Blob Animation Video Generation Guide

## Issue
Open-Sora requires ColossalAI, which does not support native Windows installation. You need to use one of these approaches:

---

## Option 1: Windows Subsystem for Linux (WSL) - Recommended
### Setup WSL2 with Ubuntu
```bash
# In PowerShell (Admin):
wsl --install -d Ubuntu

# Then in WSL bash:
cd /mnt/c/Users/amink/.copilot/repos/copilot-worktrees/Open-Sora/salihcamal2-art-ideal-telegram

# Install Python dependencies
sudo apt-get update
sudo apt-get install -y python3.10 python3-pip cuda-toolkit

# Install Open-Sora
pip install -e . --no-build-isolation

# Generate video
python scripts/inference.py \
  --config configs/opensora/inference/16x256x256.py \
  --prompt-path jelly_blob_prompt.txt
```

---

## Option 2: Docker Container
### Build and run Docker container
```bash
# Create Dockerfile in repo root
docker build -f Dockerfile.opensora -t opensora:latest .

# Run container
docker run --gpus all -it \
  -v C:/Users/amink/.copilot/repos/copilot-worktrees/Open-Sora:/workspace \
  opensora:latest \
  python scripts/inference.py \
    --config configs/opensora/inference/16x256x256.py \
    --prompt-path jelly_blob_prompt.txt
```

---

## Option 3: Cloud GPU Services
Use services like:
- **Google Colab** (free GPU access)
- **Paperspace Gradient**
- **Lambda Labs**
- **AWS EC2 with GPU**

Upload the repo and run the inference script.

---

## Option 4: Use Pre-trained Models Online
Use existing Open-Sora services:
- Visit: https://huggingface.co/spaces/hpcaitech/Open-Sora
- Upload your prompt and generate video

---

## Video Generation Command (Once Installed)

```bash
# Basic inference
python scripts/inference.py \
  --config configs/opensora/inference/16x256x256.py \
  --prompt-path jelly_blob_prompt.txt \
  --save-dir ./outputs/jelly_blob_video

# With custom parameters
python scripts/inference.py \
  --config configs/opensora/inference/16x256x256.py \
  --prompt "Your prompt here" \
  --num-frames 24 \
  --image-size 512 512 \
  --batch-size 1 \
  --save-dir ./outputs/jelly_blob_video
```

---

## Prompt Used for Your Video

The following prompt was prepared:

```
3D animated children's cartoon, Pixar-quality render, glossy squishy gelatin surfaces with soft inner glow, extremely simple rounded shapes designed for toddlers, vibrant saturated candy colors, bright cheerful lighting, smooth clean composition. WOBBO: bright yellow translucent jelly blob kid, round teardrop shape, big glossy eyes, tiny stubby arms, bouncy soft body with inner glow, extremely cute. DOT: tiny purple grape-size baby jelly blob, smallest family member, shiny purple with sparkle particles, giggly. Fast playful pacing, DOT giggles 'Boo!' in center frame, then pops and vanishes in a tiny purple sparkle burst with star particles. WOBBO gasps, his whole jelly body wobbles backward in comic shock, eyes growing huge to 150% size. Soft bounce squash and stretch animation. Fast playful surprise mood. House location. Bright cheerful toddler-friendly composition.
```

This prompt has been saved to: `jelly_blob_prompt.txt`

---

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA (RTX 3090, A100, H100 recommended)
- **RAM**: 32GB+ (preferably 64GB)
- **Disk**: 100GB+ for models and outputs
- **Time**: 2-5 minutes per video (varies by resolution/frames)

---

## Next Steps

1. Choose installation method above (WSL recommended for Windows users)
2. Follow setup instructions for your choice
3. Run the video generation command
4. Video will be saved to `./outputs/jelly_blob_video/`

Generated videos will include:
- MP4 format for compatibility
- GIF preview
- Frame quality: 256x256 to 512x512
- Duration: 16-24 frames at 8-24 fps
