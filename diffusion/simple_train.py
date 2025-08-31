import os
import sys
import mlx.core as mx
import mlx.optimizers as optim

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.simple_model import SimpleUNet
from data.cifar10 import load_train_data, CIFAR10DataLoader


def add_noise(images, noise_level=0.1):
    """Add Gaussian noise to images"""
    noise = mx.random.normal(images.shape) * noise_level
    return images + noise


def train_step(model, optimizer, clean_images):
    """Single training step - train to denoise"""
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
    print("Simple UNet Training for CIFAR-10")

    # Load data
    print("Loading CIFAR-10...")
    train_images, train_labels = load_train_data(
        data_dir="./cifar-10", download=True, normalize=True
    )

    # Use only 100 images for testing
    train_images = train_images[:100]
    train_labels = train_labels[:100]

    # Create dataloader
    train_loader = CIFAR10DataLoader(
        train_images, train_labels, batch_size=4, shuffle=True
    )

    print(f"Dataset: {len(train_images)} images")
    print(f"Image shape: {train_images.shape[1:]}")

    # Create model and optimizer
    model = SimpleUNet(in_channels=3, out_channels=3)
    optimizer = optim.Adam(learning_rate=1e-3)

    # Test single batch
    print("\nTesting single batch...")
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")

    try:
        loss = train_step(model, optimizer, images)
        print(f"Loss: {loss.item():.4f}")
        print("✓ Training step successful!")

        # Train for a few iterations
        print("\nTraining for 10 steps...")
        for i in range(10):
            images, labels = next(iter(train_loader))
            loss = train_step(model, optimizer, images)
            if i % 2 == 0:
                print(f"Step {i}: Loss = {loss.item():.4f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
