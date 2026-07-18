# 🎬 JELLY BLOB VIDEO GENERATION - QUICK START

## ⚡ Fastest Option: Google Colab (Free GPU, No Setup!)

1. Open: https://colab.research.google.com
2. Create new notebook
3. Copy-paste this code:

```python
# Install Open-Sora
!git clone https://github.com/hpcaitech/Open-Sora.git
%cd Open-Sora
!pip install -e . -q

# Create prompt
prompt = """3D animated children's cartoon, Pixar-quality render, glossy squishy gelatin 
surfaces with soft inner glow, extremely simple rounded shapes designed for toddlers, 
vibrant saturated candy colors, bright cheerful lighting. WOBBO: bright yellow translucent 
jelly blob kid, round teardrop shape, big glossy eyes, tiny stubby arms, bouncy soft body 
with inner glow, extremely cute. DOT: tiny purple grape-size baby jelly blob, shiny purple 
with sparkle particles, giggly. Fast playful pacing, DOT giggles 'Boo!', pops in tiny 
purple sparkle burst. WOBBO gasps, body wobbles backward, eyes huge 150% size. Soft bounce 
squash and stretch. Fast playful surprise mood. House location. Bright cheerful."""

with open("prompt.txt", "w") as f:
    f.write(prompt)

# Generate video (this will download models automatically)
!python scripts/inference.py \
  --config configs/opensora/inference/16x256x256.py \
  --prompt-path prompt.txt \
  --save-dir ./outputs

# Download result
from google.colab import files
files.download('outputs/sample_0.mp4')
```

4. Run all cells
5. Video downloads automatically! 🎉

---

## 🖥️ For Windows Users: WSL Method

```bash
# 1. Install WSL2 (PowerShell as Admin)
wsl --install -d Ubuntu

# 2. In Ubuntu terminal
cd /mnt/c/Users/amink/.copilot/repos/copilot-worktrees/Open-Sora/salihcamal2-art-ideal-telegram
bash setup_wsl.sh

# 3. Generate video
source ~/opensora-env/bin/activate
python scripts/inference.py --config configs/opensora/inference/16x256x256.py --prompt-path jelly_blob_prompt.txt
```

---

## 📊 Generation Time Estimates

| Method | Setup Time | Gen Time | Quality | Cost |
|--------|-----------|----------|---------|------|
| Colab | 5 min | 3 min | 256×256 | FREE |
| WSL | 30 min | 3 min | 256×256 | Local GPU |
| Docker | 15 min | 3 min | 256×256 | Local GPU |
| Cloud GPU | 5 min | 2 min | 512×512 | $5-20 |

---

## 📁 Files Created

```
✅ jelly_blob_prompt.txt              - Your animation prompt
✅ GENERATE_VIDEO_README.md            - Complete guide
✅ JELLY_BLOB_VIDEO_GENERATION.md      - Detailed setup
✅ setup_wsl.sh                        - WSL installation
✅ run_docker.sh                       - Docker setup
✅ download_models.sh                  - Model downloader
✅ Dockerfile.opensora                 - Docker config
✅ generate_jelly_blob_video.py        - Python helper
```

---

## 🎯 Recommended: Try Colab First

- No installation
- Free GPU
- 30 minutes to first video
- Perfect for testing

Then use WSL/Docker for production videos.

---

## 📞 Need Help?

- **Colab Issues**: Check GPU availability (Runtime > Change runtime > GPU)
- **WSL Issues**: Ensure WSL2 is installed and updated
- **Docker Issues**: Verify GPU drivers with `nvidia-smi`
- **Models Missing**: Run `bash download_models.sh`

---

**Your video is ready to generate! 🚀**

Choose your method above and follow the steps.
