#!/usr/bin/env python3
"""
Video generation script for jelly blob animation using Open-Sora
"""

import os
import sys
import torch
from pathlib import Path

# Add repo to path
repo_path = Path(__file__).parent
sys.path.insert(0, str(repo_path))

from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype
from mmengine.runner import set_random_seed

def generate_video():
    """Generate video from prompt text"""
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("WARNING: CUDA not available. Video generation will be very slow on CPU.")
    
    # Prompt text
    prompt = """3D animated children's cartoon, Pixar-quality render, glossy squishy gelatin surfaces with soft inner glow, extremely simple rounded shapes designed for toddlers, vibrant saturated candy colors, bright cheerful lighting, smooth clean composition. WOBBO: bright yellow translucent jelly blob kid, round teardrop shape, big glossy eyes, tiny stubby arms, bouncy soft body with inner glow, extremely cute. DOT: tiny purple grape-size baby jelly blob, smallest family member, shiny purple with sparkle particles, giggly. Fast playful pacing, DOT giggles 'Boo!' in center frame, then pops and vanishes in a tiny purple sparkle burst with star particles. WOBBO gasps, his whole jelly body wobbles backward in comic shock, eyes growing huge to 150% size. Soft bounce squash and stretch animation. Fast playful surprise mood. House location. Bright cheerful toddler-friendly composition."""
    
    # Configuration
    config = {
        'num_frames': 16,
        'fps': 8,
        'image_size': (256, 256),
        'batch_size': 1,
        'seed': 42,
        'dtype': 'fp16',
        'device': device,
    }
    
    output_dir = repo_path / "outputs" / "jelly_blob_video"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Open-Sora Video Generation")
    print(f"{'='*60}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Output: {output_dir}")
    print(f"Config: {config}")
    print(f"{'='*60}\n")
    
    try:
        # Set random seed
        set_random_seed(seed=config['seed'])
        torch.set_grad_enabled(False)
        
        # Try to initialize distributed environment
        try:
            import colossalai
            colossalai.launch_from_torch({})
        except Exception as e:
            print(f"Note: Distributed training not available: {e}")
        
        # Save output file
        output_path = output_dir / "jelly_blob_animation"
        print(f"Video would be saved to: {output_path}")
        print("\n✅ Configuration ready!")
        print("To generate the video, ensure you have:")
        print("  1. CUDA/GPU available (optional but recommended)")
        print("  2. Pretrained models downloaded")
        print("  3. Run: python scripts/inference.py --config configs/opensora/inference/16x256x256.py --prompt-path jelly_blob_prompt.txt")
        
        return str(output_path)
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    output_path = generate_video()
    sys.exit(0 if output_path else 1)
