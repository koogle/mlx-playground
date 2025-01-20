import os
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pathlib import Path
import argparse
from model import UNet
from scheduler import NoiseScheduler
from loss import diffusion_loss
from data.voc import VOCDiffusionDataset, create_data_loader


def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    model_path = os.path.join(save_dir, f"diffusion_epoch_{epoch}.npz")
    model.save_weights(model_path)
    print(f"Saved model to {model_path}")

    # Save optimizer state
    optim_path = os.path.join(save_dir, f"optimizer_epoch_{epoch}.npz")
    mx.savez(
        optim_path,
        learning_rate=mx.array(optimizer.learning_rate),
        step=mx.array(optimizer.state.get("step", 0)),
    )
    print(f"Saved optimizer state to {optim_path}")


def train_step(model, scheduler, optimizer, images, text_embeddings):
    """Single training step"""
    # Print shapes for debugging
    print(f"\nDebug shapes in train_step:")
    print(f"Images shape: {images.shape}")

    # Sample random timesteps
    batch_size = images.shape[0]
    t = mx.random.randint(0, scheduler.num_timesteps, (batch_size,))
    print(f"Timesteps shape: {t.shape}")

    # Add noise to images
    noise = mx.random.normal(images.shape)
    print(f"Noise shape: {noise.shape}")

    noisy_images = scheduler.q_sample(images, t, noise=noise)
    print(f"Noisy images shape: {noisy_images.shape}")

    def loss_fn(model_params):
        # Expand timesteps to match model's expected input format
        t_expanded = mx.expand_dims(t, axis=-1)  # [batch_size, 1]
        print(f"Expanded timesteps shape: {t_expanded.shape}")

        # Get model prediction
        predicted_noise = model.apply(model_params, noisy_images, t_expanded)
        print(f"Predicted noise shape: {predicted_noise.shape}")

        return diffusion_loss(predicted_noise, noise)

    loss, grads = nn.value_and_grad(loss_fn)(model.parameters())
    optimizer.update(model, grads)
    mx.eval(model.parameters())

    return loss


def train_epoch(model, scheduler, train_loader, optimizer):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    start_time = time.time()

    for batch_idx, (images, descriptions) in enumerate(train_loader):
        # Training step - no try/except, let errors propagate
        loss = train_step(model, scheduler, optimizer, images, descriptions)
        current_loss = loss.item()
        total_loss += current_loss
        num_batches += 1

        # Print progress
        if batch_idx % 5 == 0:
            avg_loss = total_loss / num_batches
            print(f"\nBatch {batch_idx}/{len(train_loader)}")
            print(f"Loss: {current_loss:.4f}, Avg Loss: {avg_loss:.4f}")
            print(f"Sample text: {descriptions[0]}")
            # Print shapes for debugging
            print(f"Images shape: {images.shape}")
            print(f"Current batch size: {images.shape[0]}")

    epoch_loss = total_loss / max(num_batches, 1)
    epoch_time = time.time() - start_time

    return epoch_loss, epoch_time


def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Model Training")
    parser.add_argument(
        "--mode",
        type=str,
        default="dev",
        choices=["dev", "full"],
        help="Training mode: dev (local development) or full (full training)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="./VOCdevkit/VOC2012",
        help="Path to VOC dataset",
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
    print("\nInitializing training...")
    print(f"Mode: {args.mode}")
    print(f"Data directory: {args.data_dir}")

    # Set image size and batch size based on mode
    image_size = 64  # Start with smaller images for faster training
    batch_size = args.batch_size or (4 if args.mode == "dev" else 32)

    # Create dataset with size limits for dev mode
    print("\nCreating datasets...")
    train_dataset = VOCDiffusionDataset(
        data_dir=args.data_dir, year="2012", image_set="train", img_size=image_size
    )

    # Limit dataset size in dev mode
    if args.mode == "dev":
        dev_size = 10  # Use only 10 images for dev mode
        train_dataset.image_ids = train_dataset.image_ids[:dev_size]
        print(f"Dev mode: Limited to {dev_size} images")

    # Create data loader
    train_loader = create_data_loader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    print(f"Dataset size: {len(train_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Expected batches per epoch: {len(train_dataset) // batch_size}")

    # Initialize model, scheduler and optimizer
    model = UNet(
        in_channels=3,
        model_channels=64,  # Smaller model for dev mode
        out_channels=3,
        num_res_blocks=1,
        attention_levels=[2],
        channel_mult=(1, 2, 4),
        time_emb_dim=64,
    )

    scheduler = NoiseScheduler(
        num_timesteps=100,  # Fewer timesteps for faster training
        beta_start=1e-4,
        beta_end=0.02,
    )

    optimizer = optim.Adam(learning_rate=1e-4)

    # Verify data loader
    print("\nVerifying data loader...")
    try:
        first_batch = next(iter(train_loader))
        images, descriptions = first_batch
        print(f"First batch shapes - Images: {images.shape}")
        print(f"Sample description: {descriptions[0]}")
    except Exception as e:
        print(f"Error loading first batch: {str(e)}")
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
