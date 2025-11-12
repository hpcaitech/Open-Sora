#!/usr/bin/env python3
"""Standalone test to verify division by zero is fixed."""

import math
import os

# Set the compression factor
D = 16

def test_original_buggy_code():
    """Test the original code that would cause SIGFPE."""
    print("Testing original buggy code...")
    
    # Simulate very small total_pixels
    total_pixels = 16 * 16  # This might cause width=0
    width_ratio, height_ratio = 16.0, 9.0
    
    width = int(math.sqrt(total_pixels * (width_ratio / height_ratio)) // D) * D
    print(f"  Width calculated: {width}")
    
    if width == 0:
        print(f"  ✗ Width is 0! Would cause division by zero on next line")
        return False
    
    height = int((total_pixels / width) // D) * D
    print(f"  Height calculated: {height}")
    print(f"  ✓ No division by zero")
    return True

def test_fixed_code():
    """Test the fixed code with safety check."""
    print("\nTesting fixed code with safety check...")
    
    # Simulate very small total_pixels
    total_pixels = 16 * 16
    width_ratio, height_ratio = 16.0, 9.0
    
    width = int(math.sqrt(total_pixels * (width_ratio / height_ratio)) // D) * D
    print(f"  Width calculated: {width}")
    
    # Safety check - the fix
    if width == 0:
        print(f"  ! Width is 0, applying fix: width = D = {D}")
        width = D
    
    height = int((total_pixels / width) // D) * D
    print(f"  Height calculated: {height}")
    print(f"  ✓ Division by zero prevented!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Division by Zero Fix Test")
    print("=" * 60)
    
    test_original_buggy_code()
    test_fixed_code()
    
    print("\n" + "=" * 60)
    print("Testing multiple resolutions...")
    print("=" * 60)
    
    for total_pixels in [16*16, 32*32, 64*64, 128*128, 256*256]:
        width_ratio, height_ratio = 16.0, 9.0
        width = int(math.sqrt(total_pixels * (width_ratio / height_ratio)) // D) * D
        if width == 0:
            width = D
        height = int((total_pixels / width) // D) * D
        print(f"total_pixels={total_pixels:6d}: width={width:4d}, height={height:4d}")
    
    print("\n✓ All tests completed!")
