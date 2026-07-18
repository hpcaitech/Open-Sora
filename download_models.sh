#!/bin/bash
# Download pretrained models for Open-Sora

set -e

echo "=========================================="
echo "Open-Sora Model Downloader"
echo "=========================================="

MODELS_DIR="./pretrained_models"
mkdir -p "$MODELS_DIR"

echo "Downloading pretrained models..."
echo ""

# T5 text encoder
echo "📥 Downloading T5 text encoder..."
if [ ! -d "$MODELS_DIR/t5_ckpts" ]; then
    python -c "
from transformers import T5Tokenizer, T5EncoderModel
print('Downloading T5...')
tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
model = T5EncoderModel.from_pretrained('google/t5-v1_1-xxl')
tokenizer.save_pretrained('$MODELS_DIR/t5_ckpts')
model.save_pretrained('$MODELS_DIR/t5_ckpts')
print('✅ T5 downloaded')
    "
else
    echo "✅ T5 already exists"
fi

# VAE model (from Stability AI)
echo ""
echo "📥 Downloading VAE model..."
if [ ! -d "$MODELS_DIR/vae" ]; then
    python -c "
from diffusers import AutoencoderKL
print('Downloading VAE...')
vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')
vae.save_pretrained('$MODELS_DIR/vae')
print('✅ VAE downloaded')
    "
else
    echo "✅ VAE already exists"
fi

# Open-Sora main model
echo ""
echo "📥 Downloading Open-Sora STDiT model..."
if [ ! -f "$MODELS_DIR/opensora_512x512_v1.0.pt" ]; then
    echo "Downloading from HuggingFace..."
    wget -q -O "$MODELS_DIR/opensora_512x512_v1.0.pt" \
        https://huggingface.co/hpcaitech/Open-Sora/resolve/main/model_weights/opensora_512x512_v1.0.pt && \
    echo "✅ Open-Sora model downloaded" || \
    echo "⚠️  Manual download required - Visit: https://huggingface.co/hpcaitech/Open-Sora"
else
    echo "✅ Open-Sora model already exists"
fi

echo ""
echo "=========================================="
echo "Model download complete!"
echo "=========================================="
echo ""
echo "Models located in: $MODELS_DIR"
echo ""
echo "Ready to generate videos! 🎬"
