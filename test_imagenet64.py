#!/usr/bin/env python3
"""
Test script for ImageNet-64 dataset implementation
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.imagenet64 import load_train_data, ImageNet64DataLoader

def test_imagenet64():
    """Test ImageNet-64 dataset loading and DataLoader"""
    print("Testing ImageNet-64 implementation...")
    print("=" * 50)
    
    try:
        # Test loading data (will create synthetic data if not available)
        print("1. Loading training data...")
        train_images, train_labels = load_train_data(
            data_dir="./imagenet64-test", download=True, normalize=True
        )
        
        print(f"✅ Training data loaded successfully")
        print(f"   Shape: {train_images.shape}")
        print(f"   Labels shape: {train_labels.shape}")
        print(f"   Image range: [{train_images.min():.3f}, {train_images.max():.3f}]")
        print(f"   Unique labels: {len(set(train_labels))}")
        
        # Test DataLoader
        print("\n2. Testing DataLoader...")
        train_loader = ImageNet64DataLoader(
            train_images, train_labels, batch_size=16, shuffle=True
        )
        
        print(f"✅ DataLoader created successfully")
        print(f"   Batch size: 16")
        print(f"   Number of batches: {len(train_loader)}")
        
        # Test batch iteration
        print("\n3. Testing batch iteration...")
        for i, (batch_images, batch_labels) in enumerate(train_loader):
            print(f"   Batch {i+1}: images {batch_images.shape}, labels {batch_labels.shape}")
            if i >= 2:  # Test first 3 batches
                break
        
        print("\n✅ All tests passed!")
        print("ImageNet-64 dataset implementation is working correctly.")
        
        # Clean up test directory
        import shutil
        if os.path.exists("./imagenet64-test"):
            shutil.rmtree("./imagenet64-test")
            print("🧹 Cleaned up test files")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_imagenet64()
    sys.exit(0 if success else 1)