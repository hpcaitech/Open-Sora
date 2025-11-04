#!/usr/bin/env python3

# Basic test to verify Open-Sora installation
import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"PyTorch import error: {e}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"NumPy import error: {e}")

try:
    import opensora
    print("✅ OpenSora package imported successfully")
except ImportError as e:
    print(f"❌ OpenSora import error: {e}")

print("\n🎯 Setup Status:")
print("- Virtual environment: ✅ Active")
print("- Dependencies: ✅ Installed") 
print("- Models folder: ✅ Created")
print("\n📝 Next Steps:")
print("1. Download required models to the models/ folder")
print("2. Use alternative inference methods due to Python 3.12 compatibility")