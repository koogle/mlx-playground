"""
Flow Matching Training Script for CIFAR-10
Implements Rectified Flow for image generation

Features:
- Overfit mode on single image for testing
- Full training on CIFAR-10
- Checkpoint saving/loading
- Sample generation during training
"""

import os
import sys
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow_matching.model import FlowMatchModel
from diffusion.utils import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint_in_dir,
    create_interrupt_handler,
)
from data.cifar10 import load_train_data, CIFAR10DataLoader

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def flow_matching_loss(model, x_0, x_1, t):
    """
    Compute Flow Matching loss: ||v(x_t, t) - (x_1 - x_0)||^2

    Args:
        model: Flow matching model
        x_0: Source samples (noise for generation)
        x_1: Target samples (real images)
        t: Time values in [0, 1]
    """
    # Interpolate between x_0 and x_1
    t_expanded = t[:, None, None, None]  # Expand for broadcasting
    x_t = t_expanded * x_1 + (1 - t_expanded) * x_0

    # Target velocity is the straight path
    v_target = x_1 - x_0

    # Predict velocity
    v_pred = model(x_t, t)

    # MSE loss
    loss = mx.mean((v_pred - v_target) ** 2)

    return loss


def train_step(model, optimizer, images, overfit_mode=False, iteration=0):
    """
    Single Flow Matching training step

    Args:
        model: Flow matching model
        optimizer: Optimizer
        images: Batch of real images [batch, channels, height, width]
        overfit_mode: If True, use controlled noise sampling
        iteration: Current iteration number (unused now, kept for compatibility)
    """
    # Images come in CHW format, convert to HWC for MLX
    images = mx.transpose(images, (0, 2, 3, 1))

    # Normalize to [-1, 1]
    x_1 = images * 2.0 - 1.0

    batch_size = x_1.shape[0]

    # Sample noise
    x_0 = mx.random.normal(x_1.shape)

    @mx.compile
    def loss_fn(params):
        model.update(params)

        t = mx.random.uniform(shape=(batch_size,))

        # Compute loss
        loss = flow_matching_loss(model, x_0, x_1, t)

        return loss

    loss, grads = mx.value_and_grad(loss_fn)(model.parameters())
    optimizer.update(model, grads)
    mx.eval(loss)

    return loss


def evaluate_reconstruction(model, x_1, num_steps=100, verbose=False):
    """
    Evaluate how well the model reconstructs x_1 starting from noise.
    This is a diagnostic to check if the model learned the correct flow.
    """
    # Start from random noise
    x_0 = mx.random.normal(x_1.shape)
    x = x_0

    # Solve ODE from t=0 to t=1
    dt = 1.0 / num_steps

    # Track velocity magnitudes to debug
    velocity_norms = []

    for i in range(num_steps):
        # Current time for this step
        # We're at time t = i/num_steps and will integrate to (i+1)/num_steps
        t_curr = i / num_steps
        t = mx.full((x.shape[0],), t_curr)

        # Get velocity at current position and time
        v = model(x, t)

        # Track velocity magnitude
        v_norm = mx.sqrt(mx.mean(v**2))
        velocity_norms.append(v_norm.item())

        # Euler step
        x = x + v * dt

    # Compute reconstruction error
    mse = mx.mean((x - x_1) ** 2)

    if verbose:
        print(f"  Reconstruction MSE: {mse.item():.6f}")
        # Check if velocities are dying out
        avg_early_v = np.mean(velocity_norms[: num_steps // 4])
        avg_late_v = np.mean(velocity_norms[-num_steps // 4 :])
        print(f"  Avg velocity norm - early: {avg_early_v:.4f}, late: {avg_late_v:.4f}")
        if avg_late_v < avg_early_v * 0.1:
            print("  WARNING: Velocity appears to be dying out near t=1")

    return x, mse


def sample_images(model, num_samples=4, num_steps=100, method="euler"):
    """
    Generate images using the learned flow

    Args:
        model: Trained flow matching model
        num_samples: Number of samples to generate
        num_steps: Number of ODE solver steps (more steps = better quality)
        method: Integration method ("euler" or "midpoint")
    """

    # Start from noise x_0 ~ N(0, I)
    x = mx.random.normal((num_samples, 32, 32, 3))

    # Solve ODE from t=0 to t=1
    dt = 1.0 / num_steps

    if method == "euler":
        # Standard Euler method
        for i in range(num_steps):
            t_curr = i / num_steps
            t = mx.full((num_samples,), t_curr)
            v = model(x, t)
            x = x + v * dt
    elif method == "midpoint":
        # Midpoint method (2nd order Runge-Kutta) - more accurate
        for i in range(num_steps):
            t_curr = i / num_steps
            t = mx.full((num_samples,), t_curr)
            v1 = model(x, t)

            # Midpoint
            x_mid = x + v1 * (dt / 2)
            t_mid_val = (i + 0.5) / num_steps
            t_mid = mx.full((num_samples,), t_mid_val)
            v_mid = model(x_mid, t_mid)

            # Update using midpoint velocity
            x = x + v_mid * dt

    # Denormalize from [-1, 1] to [0, 1]
    x = (x + 1.0) / 2.0
    x = mx.clip(x, 0.0, 1.0)

    return x


def save_samples(samples, epoch, save_dir="./samples"):
    """Save generated samples as grid image"""
    os.makedirs(save_dir, exist_ok=True)

    # Convert to numpy and uint8
    samples_np = np.array(samples)
    samples_uint8 = (samples_np * 255).astype(np.uint8)

    # Create grid
    num_samples = len(samples_uint8)
    grid_size = int(np.sqrt(num_samples))
    if grid_size * grid_size < num_samples:
        grid_size += 1

    # Create grid image
    img_size = 32
    padding = 2
    grid_img = np.zeros(
        (
            grid_size * (img_size + padding) + padding,
            grid_size * (img_size + padding) + padding,
            3,
        ),
        dtype=np.uint8,
    )

    for idx, sample in enumerate(samples_uint8):
        if idx >= grid_size * grid_size:
            break
        row = idx // grid_size
        col = idx % grid_size

        y_start = row * (img_size + padding) + padding
        x_start = col * (img_size + padding) + padding

        grid_img[y_start : y_start + img_size, x_start : x_start + img_size] = sample

    # Save
    grid_path = os.path.join(save_dir, f"flow_samples_epoch_{epoch}.png")
    Image.fromarray(grid_img).save(grid_path)
    print(f"  Saved samples to {grid_path}")


def train_epoch(model, optimizer, train_loader, epoch, overfit_mode=False):
    """Train for one epoch"""
    total_loss = 0
    num_batches = 0
    batch_start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        # We don't use labels for unconditional generation
        iteration = epoch * len(train_loader) + batch_idx
        loss = train_step(model, optimizer, images, overfit_mode, iteration)
        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1

        # Print progress
        print_freq = 10 if overfit_mode else 100
        if batch_idx % print_freq == 0:
            batch_time = time.time() - batch_start_time
            avg_loss = total_loss / num_batches
            print(
                f"  Batch {batch_idx}/{len(train_loader)}: "
                f"Loss = {loss_val:.4f}, Avg = {avg_loss:.4f}, "
                f"Time = {batch_time:.2f}s"
            )
            batch_start_time = time.time()

    return total_loss / num_batches


def main():
    print("=" * 60)
    print("Flow Matching Training for CIFAR-10")
    print("(Rectified Flow Implementation)")
    print("=" * 60)

    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overfit", action="store_true", help="Overfit mode: train on single image"
    )
    parser.add_argument(
        "--overfit_idx",
        type=int,
        default=None,
        help="Specific image index to use for overfitting (random if not specified)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode: use small dataset"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=250,
        help="Number of ODE steps for sampling (more steps = better quality, try 250-1000)",
    )
    parser.add_argument(
        "--ode_method",
        type=str,
        default="euler",
        choices=["euler", "midpoint"],
        help="ODE integration method (midpoint is more accurate but slower)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (default: 32 for overfit, 128 for normal)",
    )
    args = parser.parse_args()

    # Configuration
    default_batch_size = 32 if args.overfit else 128
    config = {
        "data_dir": "./cifar-10",
        "checkpoint_dir": "./checkpoints/flow_matching",
        "sample_dir": "./samples/flow_matching",
        "batch_size": (
            args.batch_size if args.batch_size is not None else default_batch_size
        ),
        "learning_rate": 1e-4 if args.overfit else 2e-4,
        "num_epochs": 1000 if args.overfit else 100,
        "save_every": 100 if args.overfit else 5,
        "sample_every": 50 if args.overfit else 5,
        "num_ode_steps": args.num_steps,
        "ode_method": args.ode_method,
        "overfit_mode": args.overfit,
        "resume_from": None,
    }

    # Adjust paths for overfit mode
    if args.overfit:
        config["checkpoint_dir"] = "./checkpoints/flow_matching_overfit"
        config["sample_dir"] = "./samples/flow_matching_overfit"
        print("\n*** OVERFIT MODE: Training on single image ***\n")
        mx.random.seed(42)

    # Load data
    print("\nLoading CIFAR-10...")
    train_images, train_labels = load_train_data(
        data_dir=config["data_dir"], download=True, normalize=True
    )

    # Handle different modes
    if args.overfit:
        # Overfit mode: pick a random image or use specified index
        if args.overfit_idx is not None:
            random_idx = args.overfit_idx
            print(f"Using specified image at index {random_idx} for overfitting test")
        else:
            random_idx = np.random.randint(0, len(train_images))
            print(f"Using random image at index {random_idx} for overfitting test")
        single_image = train_images[random_idx : random_idx + 1]
        single_label = train_labels[random_idx : random_idx + 1]

        # Save the original image for comparison
        os.makedirs(config["sample_dir"], exist_ok=True)
        original_img = (single_image[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        original_path = os.path.join(
            config["sample_dir"],
            f"original_image_idx{random_idx}_class{single_label[0]}.png",
        )
        Image.fromarray(original_img).save(original_path)
        print(f"Saved original image to {original_path}")
        print(f"Class: {CIFAR10_CLASSES[single_label[0]]}")

        # Repeat the single image to create a reasonable dataset size
        # We want multiple batches for better training dynamics
        num_repeats = max(config["batch_size"] * 10, 320)  # At least 10 batches worth
        train_images = np.repeat(single_image, num_repeats, axis=0)
        train_labels = np.repeat(single_label, num_repeats, axis=0)
    elif args.debug:
        # Debug mode: use subset
        print("Debug mode: Using only 100 images")
        train_images = train_images[:100]
        train_labels = train_labels[:100]
        config["num_epochs"] = 5
        config["batch_size"] = 16

    # Create dataloader
    train_loader = CIFAR10DataLoader(
        train_images,
        train_labels,
        batch_size=config["batch_size"],
        shuffle=not args.overfit,  # Don't shuffle in overfit mode
    )

    print(f"Dataset: {len(train_images)} images")
    print(f"Batch size: {config['batch_size']}")
    print(f"Batches per epoch: {len(train_loader)}")

    # Create model
    print("\nInitializing Flow Matching model...")
    model = FlowMatchModel(
        input_channels=3,
        hidden_channels=64 if args.overfit else 128,
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        time_emb_dim=256,
    )

    # Count parameters
    from mlx.utils import tree_flatten

    num_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = optim.Adam(learning_rate=config["learning_rate"])

    # Auto-load latest checkpoint (skip in overfit mode)
    start_epoch = 0
    checkpoint_to_load = config["resume_from"]

    if not args.overfit:
        if not checkpoint_to_load:
            latest_checkpoint = find_latest_checkpoint_in_dir(config["checkpoint_dir"])
            if latest_checkpoint:
                print(f"\nFound existing checkpoint: {latest_checkpoint}")
                checkpoint_to_load = latest_checkpoint

        if checkpoint_to_load:
            print(f"\nLoading checkpoint from {checkpoint_to_load}...")
            try:
                start_epoch = load_checkpoint(
                    model, optimizer, checkpoint_to_load, expected_type="flow_matching"
                )
                print(f"Resumed from epoch {start_epoch}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Starting from scratch...")
    else:
        print("\nOverfit mode: Starting from scratch (no checkpoint loading)")

    # Set up interrupt handler
    current_epoch = start_epoch
    current_loss = float("inf")
    loss_history = {"epoch_losses": [], "config": config}

    interrupt_handler = create_interrupt_handler(
        model,
        optimizer,
        epoch_getter=lambda: current_epoch + 1,
        loss_getter=lambda: current_loss,
        checkpoint_dir=config["checkpoint_dir"],
        model_type="flow_matching",
        config=config,
    )
    interrupt_handler.setup()

    # Training loop
    print(f"\nStarting training from epoch {start_epoch}...")
    print(
        f"ODE solver: {config['ode_method']} method with {config['num_ode_steps']} steps"
    )
    print("Press Ctrl+C to save and exit gracefully")
    print("-" * 40)

    best_loss = float("inf")

    for epoch in range(start_epoch, config["num_epochs"]):
        current_epoch = epoch
        if interrupt_handler.should_stop:
            break

        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        start_time = time.time()

        # Train one epoch
        avg_loss = train_epoch(
            model,
            optimizer,
            train_loader,
            epoch,
            overfit_mode=args.overfit,
        )

        # Track loss
        loss_history["epoch_losses"].append(avg_loss)
        current_loss = avg_loss

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")
        print(f"Average loss: {avg_loss:.6f}")

        # Save checkpoint (skip in overfit mode)
        if not args.overfit:
            if (epoch + 1) % config["save_every"] == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    avg_loss,
                    config["checkpoint_dir"],
                    model_type="flow_matching",
                    config=config,
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
                    model_type="flow_matching",
                    config=config,
                )
                print(f"New best model saved (loss: {best_loss:.6f})")

        if interrupt_handler.should_stop:
            break

    # Final message
    if interrupt_handler.interrupted:
        print("\nTraining interrupted but progress saved!")
    else:
        print("\n" + "=" * 60)
        print("Training completed!")

    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}")
    print(f"Samples saved to: {config['sample_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
