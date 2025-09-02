#!/usr/bin/env python3
"""
Example script showing how to use ImageNet-64 dataset with diffusion models
"""
import os
import sys
import mlx.core as mx
import mlx.optimizers as optim

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.simple_model import SimpleUNet
from data.imagenet64 import load_train_data, ImageNet64DataLoader


def add_noise(images, noise_level=0.1):
    """Add Gaussian noise to images"""
    noise = mx.random.normal(images.shape) * noise_level
    return images + noise


def train_step_imagenet64(model, optimizer, clean_images):
    """Single training step for ImageNet-64 images (64x64)"""
    # Images come in CHW format, convert to HWC for MLX
    clean_images = mx.transpose(clean_images, (0, 2, 3, 1))
    
    # Normalize to [-1, 1]
    clean_images = clean_images * 2.0 - 1.0
    
    # Add noise
    noise_level = mx.random.uniform(shape=(1,), low=0.1, high=0.9)
    noisy_images = add_noise(clean_images, noise_level)
    
    def loss_fn(params):
        model.update(params)
        predicted = model(noisy_images)
        # Simple L2 loss
        loss = mx.mean((predicted - clean_images) ** 2)
        return loss
    
    loss, grads = mx.value_and_grad(loss_fn)(model.parameters())
    optimizer.update(model, grads)
    mx.eval(loss)
    
    return loss


def main():
    print("ImageNet-64 Diffusion Training Example")
    print("=" * 50)
    
    # Load ImageNet-64 data
    print("Loading ImageNet-64 data...")
    try:
        train_images, train_labels = load_train_data(
            data_dir="./imagenet64", download=True, normalize=True
        )
        
        # Use only a subset for this example
        subset_size = min(100, len(train_images))
        train_images = train_images[:subset_size]
        train_labels = train_labels[:subset_size]
        
        print(f"✅ Loaded {len(train_images)} ImageNet-64 samples")
        print(f"   Image shape: {train_images.shape[1:]} (64x64x3)")
        print(f"   Classes: {len(set(train_labels))}")
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Create dataloader
    train_loader = ImageNet64DataLoader(
        train_images, train_labels, batch_size=4, shuffle=True
    )
    
    # Note: SimpleUNet was designed for 32x32 CIFAR-10 images
    # For 64x64 ImageNet-64, you may want to modify the model architecture
    print("\n⚠️  Note: SimpleUNet was designed for 32x32 images.")
    print("   For best results with 64x64 ImageNet-64, consider:")
    print("   - Modifying the model architecture") 
    print("   - Adding more downsampling layers")
    print("   - Adjusting the feature dimensions")
    
    # Create model and optimizer (keeping simple for demo)
    model = SimpleUNet(in_channels=3, out_channels=3)
    optimizer = optim.Adam(learning_rate=1e-4)  # Lower LR for larger images
    
    # Test single batch
    print("\nTesting single batch...")
    try:
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        
        # Note: This will likely need model architecture changes for 64x64
        loss = train_step_imagenet64(model, optimizer, images)
        print(f"✅ Training step successful! Loss: {loss.item():.4f}")
        
        print("\nTraining for a few steps...")
        for i in range(5):
            images, labels = next(iter(train_loader))
            loss = train_step_imagenet64(model, optimizer, images)
            print(f"   Step {i+1}: Loss = {loss.item():.4f}")
            
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("   This is expected if using SimpleUNet with 64x64 images")
        print("   The model architecture needs to be adapted for larger images")
    
    print("\n" + "=" * 50)
    print("ImageNet-64 dataset integration completed!")
    print("To use in your diffusion training:")
    print("  from data.imagenet64 import load_train_data, ImageNet64DataLoader")


if __name__ == "__main__":
    main()