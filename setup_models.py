#!/usr/bin/env python3
"""
Open-Sora 2.0 Model Download Script
Downloads all required models for video+audio generation
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, filepath):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file, tqdm(
        desc=filepath.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))

def setup_model_directories():
    """Create model directory structure"""
    models_dir = Path("models")
    subdirs = [
        "opensora",
        "flux", 
        "vae",
        "text_encoders",
        "audio"
    ]
    
    for subdir in subdirs:
        (models_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return models_dir

def main():
    print("🚀 Setting up Open-Sora 2.0 Models")
    print("=" * 50)
    
    models_dir = setup_model_directories()
    
    # Model URLs (these would be the actual URLs from Open-Sora documentation)
    models = {
        "Open-Sora 2.0 Main Model": {
            "url": "https://huggingface.co/hpcai-tech/Open-Sora-2.0/resolve/main/Open_Sora_v2.safetensors",
            "path": models_dir / "opensora" / "Open_Sora_v2.safetensors",
            "size": "~23.8GB"
        },
        "Flux Text-to-Image": {
            "url": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors", 
            "path": models_dir / "flux" / "flux1-dev.safetensors",
            "size": "~23.8GB"
        },
        "HunyuanVideo VAE": {
            "url": "https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan_video_vae_bf16.safetensors",
            "path": models_dir / "vae" / "hunyuan_vae.safetensors", 
            "size": "~2.3GB"
        }
    }
    
    print("📋 Models to download:")
    for name, info in models.items():
        print(f"  • {name}: {info['size']}")
    
    print(f"\n💾 Total download size: ~50GB")
    print(f"📁 Models will be saved to: {models_dir.absolute()}")
    
    # Note: Actual downloading would require valid URLs and proper authentication
    print("\n⚠️  Note: This script shows the structure.")
    print("📖 Please refer to Open-Sora documentation for actual model URLs and download instructions.")
    print("🔗 Visit: https://github.com/hpcaitech/Open-Sora")

if __name__ == "__main__":
    main()