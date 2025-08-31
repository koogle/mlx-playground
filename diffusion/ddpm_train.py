"""
Train DDPM on CIFAR-10
"""

import os
import sys
import time
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from pathlib import Path
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.ddpm_model import DDPM_UNet
from diffusion.scheduler import NoiseScheduler
from diffusion.utils import save_checkpoint, load_checkpoint
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

    @mx.compile
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


def sample_images(model, scheduler, num_samples=4):
    """
    Generate new images by sampling from noise
    Uses the reverse diffusion process from the scheduler
    """
    # Use scheduler's sampling method
    # Note: scheduler expects CHW format internally
    samples = scheduler.sample(model, image_size=32, batch_size=num_samples, channels=3)

    # Get the final denoised image (last in the list)
    final_images = samples[-1]

    # Convert from CHW to HWC for consistency
    final_images = mx.transpose(final_images, (0, 2, 3, 1))

    # Denormalize from [-1, 1] to [0, 1]
    final_images = (final_images + 1.0) / 2.0
    final_images = mx.clip(final_images, 0.0, 1.0)

    return final_images


def save_samples(samples, epoch, save_dir="./samples"):
    """Save generated samples as PNG images"""
    os.makedirs(save_dir, exist_ok=True)

    # Convert to numpy
    samples_np = np.array(samples)

    # Convert to uint8 (0-255 range)
    samples_uint8 = (samples_np * 255).astype(np.uint8)

    # Save each sample as a separate PNG
    for i, sample in enumerate(samples_uint8):
        save_path = os.path.join(save_dir, f"samples_epoch_{epoch}_img_{i}.png")
        img = Image.fromarray(sample)
        img.save(save_path)

    # Also create a grid image with all samples
    grid_size = int(np.sqrt(len(samples_uint8)))
    if grid_size * grid_size < len(samples_uint8):
        grid_size += 1

    # Create grid image
    img_size = samples_uint8.shape[1]
    grid_img = np.zeros((grid_size * img_size, grid_size * img_size, 3), dtype=np.uint8)

    for idx, sample in enumerate(samples_uint8):
        row = idx // grid_size
        col = idx % grid_size
        grid_img[
            row * img_size : (row + 1) * img_size, col * img_size : (col + 1) * img_size
        ] = sample

    grid_path = os.path.join(save_dir, f"samples_epoch_{epoch}_grid.png")
    Image.fromarray(grid_img).save(grid_path)

    # Print statistics
    print(f"  Generated {len(samples_np)} samples")
    print(f"  Sample range: [{samples_np.min():.3f}, {samples_np.max():.3f}]")
    print(f"  Saved to {save_dir}/samples_epoch_{epoch}_*.png")


def train_epoch(model, scheduler, optimizer, train_loader, epoch, sample_every=10):
    """Train for one epoch"""
    total_loss = 0
    num_batches = 0
    batch_start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        # Training step
        loss = train_step(model, scheduler, optimizer, images)
        total_loss += loss.item()
        num_batches += 1

        # Print progress
        if batch_idx % 100 == 0:
            batch_time = time.time() - batch_start_time
            avg_loss = total_loss / num_batches
            print(
                f"  Batch {batch_idx}/{len(train_loader)}: Loss = {loss.item():.4f}, Avg = {avg_loss:.4f}, Time = {batch_time:.2f}s"
            )
            batch_start_time = time.time()  # Reset timer for next batch group

    return total_loss / num_batches


def main():
    print("=" * 60)
    print("DDPM Training for CIFAR-10")
    print("=" * 60)

    # Configuration
    config = {
        "data_dir": "./cifar-10",
        "checkpoint_dir": "./checkpoints/ddpm",
        "sample_dir": "./samples/ddpm",
        "batch_size": 32,
        "learning_rate": 2e-4,
        "num_epochs": 100,
        "num_timesteps": 1000,
        "sample_every": 50,  # Sample every N batches
        "save_every": 5,  # Save checkpoint every N epochs
        "resume_from": None,  # Set to checkpoint path to resume
    }

    # Load data
    print("\nLoading CIFAR-10...")
    train_images, train_labels = load_train_data(
        data_dir=config["data_dir"], download=True, normalize=True
    )

    # For testing, use subset
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        print("Debug mode: Using only 1000 images")
        train_images = train_images[:1000]
        train_labels = train_labels[:1000]
        config["num_epochs"] = 5

    # Create dataloader
    train_loader = CIFAR10DataLoader(
        train_images, train_labels, batch_size=config["batch_size"], shuffle=True
    )

    print(f"Dataset: {len(train_images)} images")
    print(f"Batch size: {config['batch_size']}")
    print(f"Batches per epoch: {len(train_loader)}")

    # Create model
    print("\nInitializing model...")
    model = DDPM_UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_multipliers=(1, 2, 2, 2),
        num_res_blocks=2,
        attention_resolutions=(8,),  # Add attention at 8x8 resolution
    )

    # Create scheduler
    scheduler = NoiseScheduler(
        num_timesteps=config["num_timesteps"],
        beta_start=1e-4,
        beta_end=0.02,
    )

    # Create optimizer
    optimizer = optim.Adam(learning_rate=config["learning_rate"])

    # Load checkpoint if resuming
    start_epoch = 0
    if config["resume_from"]:
        print(f"\nLoading checkpoint from {config['resume_from']}...")
        try:
            start_epoch = load_checkpoint(model, optimizer, config["resume_from"])
            print(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting from scratch...")

    # Training loop
    print(f"\nStarting training from epoch {start_epoch}...")
    print("-" * 40)

    best_loss = float("inf")

    for epoch in range(start_epoch, config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        start_time = time.time()

        # Train one epoch
        avg_loss = train_epoch(
            model,
            scheduler,
            optimizer,
            train_loader,
            epoch,
            sample_every=config["sample_every"],
        )

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")
        print(f"Average loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % config["save_every"] == 0:
            save_checkpoint(
                model, optimizer, epoch + 1, avg_loss, config["checkpoint_dir"]
            )

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                avg_loss,
                os.path.join(config["checkpoint_dir"], "best"),
            )
            print(f"New best model saved (loss: {best_loss:.4f})")

        # Generate final samples for this epoch
        print(f"\nGenerating epoch {epoch + 1} samples...")
        samples = sample_images(model, scheduler, num_samples=8)
        save_samples(samples, epoch + 1, config["sample_dir"])

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}")
    print(f"Samples saved to: {config['sample_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
