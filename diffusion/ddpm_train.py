"""
Train DDPM on CIFAR-10
"""

import os
import sys
import mlx.core as mx
import mlx.optimizers as optim

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.ddpm_model import DDPM_UNet
from diffusion.scheduler import NoiseScheduler
from data.cifar10 import load_train_data, CIFAR10DataLoader


def train_step(model, scheduler, optimizer, images):
    """Single DDPM training step"""
    # Images come in CHW format, convert to HWC for MLX
    images = mx.transpose(images, (0, 2, 3, 1))

    # Normalize to [-1, 1]
    images = images * 2.0 - 1.0

    batch_size = images.shape[0]

    # Sample random time steps
    t = mx.random.randint(0, scheduler.num_timesteps, (batch_size,))

    # Sample noise
    noise = mx.random.normal(images.shape)

    # Add noise to images (forward diffusion process)
    noisy_images = scheduler.add_noise(images, t, noise)

    def loss_fn(params):
        model.update(params)
        # Predict the noise
        predicted_noise = model(noisy_images, t)
        # L2 loss between predicted and actual noise
        loss = mx.mean((predicted_noise - noise) ** 2)
        return loss

    loss, grads = mx.value_and_grad(loss_fn)(model.parameters())
    optimizer.update(model, grads)
    mx.eval(loss)

    return loss


def main():
    print("DDPM Training for CIFAR-10")
    print("-" * 40)

    # Load data
    print("Loading CIFAR-10...")
    train_images, train_labels = load_train_data(
        data_dir="./cifar-10", download=True, normalize=True
    )

    # Use subset for testing
    num_samples = 100
    train_images = train_images[:num_samples]
    train_labels = train_labels[:num_samples]

    # Create dataloader
    batch_size = 4
    train_loader = CIFAR10DataLoader(
        train_images, train_labels, batch_size=batch_size, shuffle=True
    )

    print(f"Dataset: {len(train_images)} images")
    print(f"Batch size: {batch_size}")
    print(f"Image shape: {train_images.shape[1:]}")

    # Create model
    print("\nInitializing model...")
    model = DDPM_UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,  # Smaller model for testing
        channel_multipliers=(1, 2, 2),  # 3 levels instead of 4
        num_res_blocks=1,  # Fewer blocks for speed
        attention_resolutions=(),  # No attention for now
    )

    # Create scheduler
    scheduler = NoiseScheduler(
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
    )

    # Create optimizer
    optimizer = optim.Adam(learning_rate=2e-4)

    # Test single batch
    print("\nTesting single batch...")
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")

    try:
        loss = train_step(model, scheduler, optimizer, images)
        print(f"Initial loss: {loss.item():.4f}")
        print("✓ Training step successful!")

        # Train for a few steps
        print("\nTraining for 20 steps...")
        for i in range(20):
            images, labels = next(iter(train_loader))
            loss = train_step(model, scheduler, optimizer, images)
            if i % 5 == 0:
                print(f"Step {i:3d}: Loss = {loss.item():.4f}")

        print("\n✓ Training completed successfully!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
