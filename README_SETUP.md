# Open-Sora 2.0 Complete Setup Guide

## 🎯 Current Status
✅ **Environment Setup Complete**
- Virtual environment: `sora-env` 
- Dependencies installed with CPU compatibility
- OpenSora package installed
- Models directory created

## 🚨 Python 3.12 Compatibility Issue
The current setup has a known issue with Python 3.12 and PyTorch's Dynamo. For full functionality, consider using Python 3.10 or 3.11.

## 📁 Directory Structure
```
~/Open-Sora-All/
├── sora-env/              # Virtual environment
├── models/                # Model files (download required)
├── opensora/              # Source code
├── scripts/               # Inference scripts
├── configs/               # Configuration files
└── requirements_fixed.txt # Fixed dependencies
```

## 🔽 Model Download Requirements

### Core Models (~50GB total)
1. **Open-Sora 2.0 Main Model** (~23.8GB)
   - File: `Open_Sora_v2.safetensors`
   - Location: `models/opensora/`

2. **Flux Text-to-Image Model** (~23.8GB)
   - File: `flux1-dev.safetensors`
   - Location: `models/flux/`

3. **Video Autoencoder Models** (~2.3GB)
   - HunyuanVideo VAE: `hunyuan_vae.safetensors`
   - Location: `models/vae/`

4. **Text Encoders**
   - T5-XXL model files
   - CLIP model files
   - Location: `models/text_encoders/`

### Audio/Voice Models (for full video+audio generation)
5. **Audio Synthesis Models**
   - Text-to-speech models
   - Audio synchronization models
   - Location: `models/audio/`

## 🚀 Usage Instructions

### 1. Activate Environment
```bash
cd ~/Open-Sora-All
source sora-env/bin/activate
```

### 2. Download Models
```bash
python setup_models.py  # Shows structure and requirements
```

### 3. Basic Video Generation (once models are downloaded)
```bash
# Alternative method due to Python 3.12 compatibility
python -c "
import opensora
from opensora.models import create_model
# Custom inference code here
"
```

### 4. Video + Audio Generation
For full text-to-video with synchronized audio:
```bash
# Example command (adjust based on actual API)
python generate_video_with_audio.py \
  --prompt "A sunrise over calm ocean with gentle waves" \
  --voice "calm narrator voice" \
  --duration 5 \
  --resolution 1024x576 \
  --output sunrise_with_voice.mp4
```

## 🔧 Configuration Options

### Video Settings
- **Resolution**: 256px, 768px, 1024px options
- **Duration**: 2s to 15s
- **Aspect Ratios**: Any ratio supported
- **Frame Rate**: 24fps default

### Audio Settings  
- **Voice Types**: Multiple TTS voices available
- **Audio Quality**: High-fidelity synthesis
- **Synchronization**: Automatic lip-sync for characters
- **Background Music**: Optional ambient audio

## 📖 Next Steps

1. **Download Models**: Follow Open-Sora documentation for model downloads
2. **Python Version**: Consider using Python 3.10/3.11 for full compatibility
3. **GPU Setup**: For faster generation, configure CUDA if available
4. **Custom Configs**: Modify config files in `configs/` for specific needs

## 🔗 Resources
- [Open-Sora GitHub](https://github.com/hpcaitech/Open-Sora)
- [Model Downloads](https://huggingface.co/hpcai-tech)
- [Documentation](https://hpcaitech.github.io/Open-Sora/)

## 🆘 Troubleshooting
- **Import Errors**: Ensure virtual environment is activated
- **Model Not Found**: Check model file paths and downloads
- **Memory Issues**: Use smaller resolutions or shorter durations
- **Python 3.12 Issues**: Consider downgrading to Python 3.10/3.11