# 🎬 Jelly Blob Animation - Video Generation

Your Pixar-quality jelly blob animation has been prepared for video generation using Open-Sora!

## 📋 What's Ready

✅ **Prompt prepared**: `jelly_blob_prompt.txt` - Detailed description of the animation scene
✅ **Video configurations**: Optimized inference configs for 256x256 and 512x512 output
✅ **Setup guides**: Multiple installation options for Windows, WSL, and Docker

## 🚀 Quick Start

### 💻 **Option 1: Windows Subsystem for Linux (WSL) - Recommended**

**1. Install WSL2 in PowerShell (Admin):**
```powershell
wsl --install -d Ubuntu
```

**2. Open Ubuntu WSL terminal and run:**
```bash
cd /mnt/c/Users/amink/.copilot/repos/copilot-worktrees/Open-Sora/salihcamal2-art-ideal-telegram
bash setup_wsl.sh
```

**3. Generate video:**
```bash
source ~/opensora-env/bin/activate
python scripts/inference.py \
  --config configs/opensora/inference/16x256x256.py \
  --prompt-path jelly_blob_prompt.txt \
  --save-dir ./outputs/jelly_blob_video
```

---

### 🐳 **Option 2: Docker**

**1. Build and run Docker container:**
```bash
cd path/to/Open-Sora
bash run_docker.sh
```

**Requirements:**
- Docker installed
- Docker GPU support (`nvidia-docker`)
- 30GB disk space

---

### ☁️ **Option 3: Google Colab (Free GPU)**

Create a new Colab notebook and run:
```python
# Clone repo
!git clone https://github.com/hpcaitech/Open-Sora.git
%cd Open-Sora

# Install dependencies
!pip install -e . -q

# Download pretrained models
!wget https://huggingface.co/hpcaitech/Open-Sora/resolve/main/model_weights/opensora.pt -O pretrained_models/opensora.pt

# Create prompt file
!echo "YOUR_PROMPT_HERE" > prompt.txt

# Generate video
!python scripts/inference.py --config configs/opensora/inference/16x256x256.py --prompt-path prompt.txt
```

---

### 🌐 **Option 4: Use Online Service**

Visit: https://huggingface.co/spaces/hpcaitech/Open-Sora
- No installation required
- Free GPU access
- Simple web interface

---

## 📝 Your Animation Prompt

The prompt prepared for your jelly blob animation:

```
3D animated children's cartoon, Pixar-quality render, glossy squishy gelatin surfaces 
with soft inner glow, extremely simple rounded shapes designed for toddlers, vibrant 
saturated candy colors, bright cheerful lighting, smooth clean composition. 

WOBBO: bright yellow translucent jelly blob kid, round teardrop shape, big glossy eyes, 
tiny stubby arms, bouncy soft body with inner glow, extremely cute. 

DOT: tiny purple grape-size baby jelly blob, smallest family member, shiny purple with 
sparkle particles, giggly. 

Fast playful pacing, DOT giggles 'Boo!' in center frame, then pops and vanishes in a 
tiny purple sparkle burst with star particles. WOBBO gasps, his whole jelly body wobbles 
backward in comic shock, eyes growing huge to 150% size. Soft bounce squash and stretch 
animation. Fast playful surprise mood. House location. Bright cheerful toddler-friendly 
composition.
```

---

## ⚙️ Video Generation Parameters

Default configuration (16x256x256.py):
- **Duration**: 16 frames
- **Resolution**: 256×256 pixels
- **FPS**: 8 fps (adjustable)
- **Quality**: 2 seconds video
- **Time**: ~2-3 minutes generation

For higher quality (64x512x512.py):
- **Resolution**: 512×512 pixels  
- **Duration**: 2-4 seconds
- **Time**: ~5-10 minutes
- **GPU**: Requires high-end GPU (RTX 3090, A100)

---

## 📊 Hardware Requirements

| Component | Minimum | Recommended | Ideal |
|-----------|---------|-------------|-------|
| GPU | RTX 2080 | RTX 3090 | RTX 4090/A100 |
| VRAM | 8GB | 24GB | 40GB+ |
| System RAM | 16GB | 32GB | 64GB |
| Disk Space | 50GB | 100GB | 200GB |
| Generation Time | 10-15 min | 2-5 min | <2 min |

---

## 📁 Output Files

After generation, you'll find:

```
outputs/jelly_blob_video/
├── sample_0.mp4        # Main video file
├── sample_0.gif        # Preview GIF
├── sample_0_frames/    # Individual frames (optional)
└── sample_0_info.txt   # Generation metadata
```

---

## 🔧 Custom Generation Commands

**Higher resolution (512×512):**
```bash
python scripts/inference.py \
  --config configs/opensora/inference/64x512x512.py \
  --prompt-path jelly_blob_prompt.txt
```

**Longer video (more frames):**
```bash
python scripts/inference.py \
  --config configs/opensora/inference/16x256x256.py \
  --prompt-path jelly_blob_prompt.txt \
  --num-frames 24
```

**Custom settings:**
```bash
python scripts/inference.py \
  --config configs/opensora/inference/16x256x256.py \
  --prompt "your custom prompt here" \
  --batch-size 1 \
  --seed 42 \
  --num-sampling-steps 50
```

---

## 🐛 Troubleshooting

### "ColossalAI not found"
→ Use WSL, Docker, or Colab instead of native Windows

### "CUDA out of memory"  
→ Reduce resolution or batch size:
```bash
--image-size 256 256 --batch-size 1
```

### "Model weights not found"
→ Download from HuggingFace:
```bash
python -c "from opensora.registry import MODELS; MODELS['STDiT-XL/2'].from_pretrained('hpcaitech/Open-Sora')"
```

### "Slow generation on CPU"
→ Install GPU drivers and CUDA toolkit for 10x+ speedup

---

## 📚 Additional Resources

- **Open-Sora Docs**: https://github.com/hpcaitech/Open-Sora/tree/main/docs
- **Model Weights**: https://huggingface.co/hpcaitech/Open-Sora
- **Community**: https://github.com/hpcaitech/Open-Sora/discussions

---

## ✨ Next Steps

1. **Choose installation method** (WSL recommended for Windows)
2. **Run setup script** for your chosen method
3. **Generate video** using commands above
4. **Enjoy your jelly blob animation!** 🎉

---

Generated on: 2026-07-19
Prompt file: `jelly_blob_prompt.txt`
Setup guide: `JELLY_BLOB_VIDEO_GENERATION.md`
