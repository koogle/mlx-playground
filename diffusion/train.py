import os
import time
import mlx.core as mx
import mlx.optimizers as optim
from pathlib import Path
import argparse
from diffusion.model import UNet
from diffusion.scheduler import NoiseScheduler
from diffusion.loss import diffusion_loss
from diffusion.utils import save_checkpoint
from data.cifar10 import load_train_data, CIFAR10DataLoader


def train_step(model, scheduler, optimizer, images):
    """Single training step"""
    # Images are already MLX arrays from the dataloader
    # Normalize from [0, 1] to [-1, 1]
    images = images * 2.0 - 1.0

    # Convert from HWC to CHW format for the model
    images = mx.transpose(images, (0, 3, 1, 2))

    print("Debug shapes in train_step:")
    print(f"Images shape after conversion: {images.shape}")

    # Sample random timesteps
    batch_size = images.shape[0]
    t = mx.random.randint(0, scheduler.num_timesteps, (batch_size,))
    print(f"Timesteps shape: {t.shape}")

    # Add noise to images
    noise = mx.random.normal(images.shape)
    print(f"Noise shape: {noise.shape}")

    noisy_images = scheduler.q_sample(images, t, noise=noise)
    print(f"Noisy images shape: {noisy_images.shape}")

    def loss_fn(params):
        # Expand timesteps to match model's expected input format
        t_expanded = mx.expand_dims(t, axis=-1)  # [batch_size, 1]

        # Get model prediction
        model.update(params)
        predicted_noise = model(noisy_images, t_expanded)

        return diffusion_loss(predicted_noise, noise)

    # Compute loss and gradients
    loss, grads = mx.value_and_grad(loss_fn)(model.parameters())
    optimizer.update(model, grads)

    # Evaluate model state and loss together for better batching
    state = [model.parameters(), optimizer.state]
    mx.eval(loss, *state)  # Batch evaluation

    # Clear intermediate values
    del grads, state

    return loss


def train_epoch(model, scheduler, train_loader, optimizer):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        # Training step
        loss = train_step(model, scheduler, optimizer, images)
        current_loss = loss.item()
        total_loss += current_loss
        num_batches += 1

        # Print progress
        if batch_idx % 5 == 0:
            avg_loss = total_loss / num_batches
            print(f"\nBatch {batch_idx}/{len(train_loader)}")
            print(f"Loss: {current_loss:.4f}, Avg Loss: {avg_loss:.4f}")
            print(f"Images shape: {images.shape}")
            print(f"Labels: {labels[:5]}")  # Show first 5 labels

    epoch_loss = total_loss / max(num_batches, 1)
    epoch_time = time.time() - start_time

    return epoch_loss, epoch_time


def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Model Training")
    parser.add_argument(
        "--mode",
        type=str,
        default="dev",
        choices=["dev", "full", "test"],
        help="Training mode: dev (local development), full (full training), or test (single step test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override default batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default="./checkpoints/diffusion",
        help="Directory to save checkpoints",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Initializing training {args.mode}...")

    batch_size = args.batch_size or (4 if args.mode == "dev" else 32)

    print("Loading CIFAR-10 dataset...")
    train_images, train_labels = load_train_data(
        data_dir="./cifar-10",
        download=True,
        normalize=True,
    )

    # Limit dataset size in dev mode
    if args.mode == "dev":
        dev_size = 100  # Use only 100 images for dev mode
        train_images = train_images[:dev_size]
        train_labels = train_labels[:dev_size]
        print(f"Dev mode: Limited to {dev_size} images")

    # Create data loader
    train_loader = CIFAR10DataLoader(
        images=train_images, labels=train_labels, batch_size=batch_size, shuffle=True
    )

    print(f"Dataset size: {len(train_images)}")
    print(f"Image shape: {train_images.shape[1:]}")
    print(f"Batch size: {batch_size}")
    print(f"Expected batches per epoch: {len(train_loader)}")

    # Initialize model, scheduler and optimizer
    model = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=64,
        num_res_blocks=2,
        channel_mult=(1, 2, 2, 2),  # 4 levels: 32->16->8->4
        time_emb_dim=256,
    )

    scheduler = NoiseScheduler(
        num_timesteps=100,  # Fewer timesteps for faster training
        beta_start=1e-4,
        beta_end=0.02,
    )

    optimizer = optim.Adam(learning_rate=1e-4)

    # Test with single batch
    print("\nTesting with single batch...")
    try:
        first_batch = next(iter(train_loader))
        images, labels = first_batch
        print(f"First batch - Images: {images.shape}, Labels: {labels.shape}")
        print(f"Sample labels: {labels[:5]}")

        # Run single training step
        print("\nRunning single training step...")
        loss = train_step(model, scheduler, optimizer, images)
        print(f"Training step successful! Loss: {loss.item():.4f}")

        if args.mode == "test":
            print("\nTest mode complete - single step successful!")
            return
    except Exception as e:
        print(f"Error in test step: {str(e)}")
        import traceback

        traceback.print_exc()
        raise

    # Training loop
    print("\nStarting training...")
    best_loss = float("inf")

    try:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

            # Training
            epoch_loss, epoch_time = train_epoch(
                model, scheduler, train_loader, optimizer
            )

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Loss: {epoch_loss:.4f}")
            print(f"Time: {epoch_time:.1f}s")

            # Save checkpoint if best loss
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_checkpoint(model, optimizer, epoch + 1, epoch_loss, args.save_dir)
                print("New best model saved!")

            # Regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    epoch_loss,
                    os.path.join(args.save_dir, "regular"),
                )
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        raise  # Re-raise the exception to see the full traceback


if __name__ == "__main__":
    main()
