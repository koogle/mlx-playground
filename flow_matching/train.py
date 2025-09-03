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
from mlx.utils import tree_flatten
import mlx.optimizers as optim
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Use non-interactive backend

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow_matching.model import FlowMatchModel
from utils.checkpoints import (
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


def flow_matching_loss(model, x_0, x_1, t, class_labels=None):
    """
    Compute Flow Matching loss: ||v(x_t, t) - (x_1 - x_0)||^2

    Args:
        model: Flow matching model
        x_0: Source samples (noise for generation)
        x_1: Target samples (real images)
        t: Time values in [0, 1]
        class_labels: Class labels for conditional generation (optional)
    """
    # Interpolate between x_0 and x_1
    t_expanded = t[:, None, None, None]  # Expand for broadcasting
    x_t = t_expanded * x_1 + (1 - t_expanded) * x_0

    # Target velocity is the straight path
    v_target = x_1 - x_0

    # Predict velocity
    v_pred = model(x_t, t, class_labels)

    # MSE loss
    loss = mx.mean((v_pred - v_target) ** 2)

    return loss


def train_step(model, optimizer, images, labels=None, overfit_mode=False, iteration=0):
    """
    Single Flow Matching training step

    Args:
        model: Flow matching model
        optimizer: Optimizer
        images: Batch of real images [batch, channels, height, width]
        labels: Batch of class labels (optional, for class-conditional generation)
        overfit_mode: If True, use controlled noise sampling
        iteration: Current iteration number (unused now, kept for compatibility)
    """
    # Images come in CHW format, convert to HWC for MLX
    images = mx.transpose(images, (0, 2, 3, 1))

    # Normalize to [-1, 1] so that center is at 0
    x_1 = images * 2.0 - 1.0

    batch_size = x_1.shape[0]

    # Sample noise from [-1, 1] range
    x_0 = mx.random.normal(x_1.shape)

    @mx.compile
    def loss_fn(params):
        model.update(params)

        t = mx.random.uniform(shape=(batch_size,))

        # Compute loss
        loss = flow_matching_loss(model, x_0, x_1, t, labels)

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
        v = model(x, t, None)  # No class conditioning for reconstruction evaluation

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


def sample_images(model, num_samples=4, num_steps=100, class_labels=None):
    """
    Generate images using the learned flow
    Following the Rectified Flow paper, we use Euler method for ODE solving.

    Args:
        model: Trained flow matching model
        num_samples: Number of samples to generate
        num_steps: Number of ODE solver steps (more steps = better quality)
        class_labels: Class labels for conditional generation (array of shape [num_samples])
    """

    # Start from noise x_0 ~ N(0, I)
    x = mx.random.normal((num_samples, 32, 32, 3))

    # Solve ODE from t=0 to t=1 using Euler method
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_curr = i / num_steps
        t = mx.full((num_samples,), t_curr)
        v = model(x, t, class_labels)
        x = x + v * dt

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


class LiveTrainingVisualizer:
    """Simple training visualizer that updates a single image file"""
    
    def __init__(self, config, overfit_mode=False, original_image=None):
        self.config = config
        self.overfit_mode = overfit_mode
        self.original_image = original_image
        
        # Training data
        self.epoch_losses = []
        self.current_batch_losses = []
        self.current_epoch = 0
        
        # Single plot file that gets updated
        self.plot_path = os.path.join(config.get("sample_dir", "./samples"), "live_training.png")
        print(f"Live training plot will be updated at: {self.plot_path}")
        
    def update_batch_loss(self, loss, batch_idx, total_batches, epoch):
        """Update with new batch loss and refresh the plot"""
        self.current_batch_losses.append(loss)
        self.current_epoch = epoch
        
        # Update plot every 5 batches to see progress
        if (batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1:
            self.update_plot()
        
    def update_epoch_complete(self, avg_loss, model):
        """Update when epoch completes"""
        self.epoch_losses.append(avg_loss)
        self.current_batch_losses = []  # Clear for next epoch
        self.update_plot(model)
        
    def update_plot(self, model=None):
        """Update the single training plot"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Flow Matching Training - Epoch {self.current_epoch + 1}", fontsize=14)
        
        # Epoch losses (top-left)
        if self.epoch_losses:
            axes[0,0].plot(range(1, len(self.epoch_losses) + 1), self.epoch_losses, 'b-', linewidth=2)
            axes[0,0].set_xlabel("Epoch")
            axes[0,0].set_ylabel("Loss")
            axes[0,0].set_title("Epoch Loss")
            axes[0,0].grid(True, alpha=0.3)
            if len(self.epoch_losses) > 1:
                axes[0,0].set_yscale("log")
        else:
            axes[0,0].text(0.5, 0.5, "No epoch data yet", ha="center", va="center", transform=axes[0,0].transAxes)
            
        # Current batch losses (top-right)
        if self.current_batch_losses:
            axes[0,1].plot(range(1, len(self.current_batch_losses) + 1), self.current_batch_losses, 'r-', alpha=0.7)
            axes[0,1].set_xlabel("Batch")
            axes[0,1].set_ylabel("Loss")
            axes[0,1].set_title(f"Current Epoch Batches ({len(self.current_batch_losses)})")
            axes[0,1].grid(True, alpha=0.3)
        else:
            axes[0,1].text(0.5, 0.5, "No batch data", ha="center", va="center", transform=axes[0,1].transAxes)
            
        # Original image (bottom-left)
        if self.overfit_mode and self.original_image is not None:
            axes[1,0].imshow(self.original_image)
            axes[1,0].set_title("Original Image")
            axes[1,0].axis("off")
        else:
            axes[1,0].text(0.5, 0.5, "Not in overfit mode", ha="center", va="center", transform=axes[1,0].transAxes)
            axes[1,0].axis("off")
            
        # Generated sample (bottom-right) - only if model provided
        if model is not None:
            if model.num_classes is not None:
                sample_class = mx.random.randint(0, model.num_classes, shape=(1,))
                sample_class_name = CIFAR10_CLASSES[sample_class.item()]
            else:
                sample_class = None
                sample_class_name = "Unconditional"
                
            sample = sample_images(model, num_samples=1, num_steps=self.config["num_ode_steps"], class_labels=sample_class)
            sample_uint8 = (np.array(sample)[0] * 255).astype(np.uint8)
            
            axes[1,1].imshow(sample_uint8)
            axes[1,1].set_title(f"Sample\n{sample_class_name}")
            axes[1,1].axis("off")
        else:
            axes[1,1].text(0.5, 0.5, "Sample at epoch end", ha="center", va="center", transform=axes[1,1].transAxes)
            axes[1,1].axis("off")
        
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        
    def close(self):
        pass


def visualize_training_progress(
    model,
    loss_history,
    epoch,
    config,
    overfit_mode=False,
    original_image=None,
    save_dir=None,
):
    """
    Create and save visualization of training progress

    Args:
        model: Current model
        loss_history: List of losses up to current epoch
        epoch: Current epoch number
        config: Configuration dict
        overfit_mode: Whether in overfit mode
        original_image: Original image for comparison (in overfit mode)
        save_dir: Directory to save visualization

    Returns:
        sample: The generated sample
    """
    # Set up plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Flow Matching Training Progress - Epoch {epoch}", fontsize=16)

    # Axes setup
    ax_loss = axes[0, 0]
    ax_sample = axes[0, 1]
    ax_original = axes[1, 0]
    ax_reconstruction = axes[1, 1]

    # Plot loss history
    if loss_history:
        ax_loss.plot(range(1, len(loss_history) + 1), loss_history, "b-", linewidth=2)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Training Loss")
        ax_loss.grid(True, alpha=0.3)
        if len(loss_history) > 1:
            ax_loss.set_yscale("log")

    # Show original image if in overfit mode
    if overfit_mode and original_image is not None:
        ax_original.imshow(original_image)
        ax_original.set_title("Original Image")
        ax_original.axis("off")
    else:
        ax_original.axis("off")
        ax_original.text(
            0.5,
            0.5,
            "Not in overfit mode",
            ha="center",
            va="center",
            transform=ax_original.transAxes,
        )

    # Generate and display sample (randomly pick a class for conditional generation)
    if model.num_classes is not None:
        sample_class = mx.random.randint(0, model.num_classes, shape=(1,))
        sample_class_name = CIFAR10_CLASSES[sample_class.item()]
    else:
        sample_class = None
        sample_class_name = "Unconditional"

    sample = sample_images(
        model,
        num_samples=1,
        num_steps=config["num_ode_steps"],
        class_labels=sample_class,
    )

    sample_np = np.array(sample)[0]
    sample_uint8 = (sample_np * 255).astype(np.uint8)

    ax_sample.imshow(sample_uint8)
    ax_sample.set_title(f"Generated Sample (Epoch {epoch})\nClass: {sample_class_name}")
    ax_sample.axis("off")

    # Show reconstruction or sample grid
    if overfit_mode and original_image is not None:
        reconstruction = sample_images(
            model,
            num_samples=1,
            num_steps=config["num_ode_steps"],
            class_labels=sample_class,
        )

        recon_np = np.array(reconstruction)[0]
        recon_uint8 = (recon_np * 255).astype(np.uint8)

        ax_reconstruction.imshow(recon_uint8)
        ax_reconstruction.set_title(f"Reconstruction (Epoch {epoch})")
        ax_reconstruction.axis("off")
    else:
        # Generate multiple samples for grid (show different classes if conditional)
        if model.num_classes is not None:
            # Show 4 different classes
            grid_classes = mx.array([0, 1, 2, 3])  # airplane, automobile, bird, cat
            grid_class_names = [CIFAR10_CLASSES[i] for i in grid_classes.tolist()]
        else:
            grid_classes = None
            grid_class_names = ["Unconditional"] * 4

        samples = sample_images(
            model,
            num_samples=4,
            num_steps=config["num_ode_steps"],
            class_labels=grid_classes,
        )

        # Create 2x2 grid
        samples_np = np.array(samples)
        grid = np.zeros((66, 66, 3))
        for i in range(2):
            for j in range(2):
                idx = i * 2 + j
                if idx < len(samples_np):
                    y_start = i * 33
                    x_start = j * 33
                    grid[y_start : y_start + 32, x_start : x_start + 32] = samples_np[
                        idx
                    ]

        grid_uint8 = (grid * 255).astype(np.uint8)
        ax_reconstruction.imshow(grid_uint8)
        if model.num_classes is not None:
            grid_title = (
                f"Sample Grid (Epoch {epoch})\nClasses: {', '.join(grid_class_names)}"
            )
        else:
            grid_title = f"Sample Grid (Epoch {epoch})"
        ax_reconstruction.set_title(grid_title)
        ax_reconstruction.axis("off")

    # Save the figure
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, f"training_progress_epoch_{epoch}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        print(f"  Saved training progress plot to {plot_path}")

    plt.close(fig)

    return sample


def train_epoch(model, optimizer, train_loader, epoch, overfit_mode=False, visualizer=None):
    """Train for one epoch"""
    total_loss = 0
    num_batches = 0
    batch_start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        iteration = epoch * len(train_loader) + batch_idx
        loss = train_step(model, optimizer, images, labels, overfit_mode, iteration)
        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1

        # Update live visualizer
        if visualizer is not None:
            visualizer.update_batch_loss(loss_val, batch_idx, len(train_loader), epoch)

        # Print progress after every batch
        batch_time = time.time() - batch_start_time
        avg_loss = total_loss / num_batches
        print(
            f"  Batch {batch_idx + 1}/{len(train_loader)}: "
            f"Loss = {loss_val:.4f}, Avg = {avg_loss:.4f}, "
            f"Time = {batch_time:.2f}s"
        )
        batch_start_time = time.time()

    return total_loss / num_batches


def train_and_visualize(
    model,
    optimizer,
    train_loader,
    config,
    start_epoch=0,
    overfit_mode=False,
    original_image=None,
    live_visualizer=None,
):
    """
    Comprehensive training function with visualization and tracking

    Args:
        model: Flow matching model
        optimizer: Optimizer
        train_loader: Data loader
        config: Configuration dict
        start_epoch: Starting epoch number
        overfit_mode: Whether in overfit mode
        original_image: Original image for comparison (in overfit mode)
        live_visualizer: Live visualization window (optional)

    Returns:
        loss_history: List of average losses per epoch
    """
    loss_history = []
    best_loss = float("inf")

    for epoch in range(start_epoch, config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        start_time = time.time()

        # Train one epoch with live visualization
        avg_loss = train_epoch(model, optimizer, train_loader, epoch, overfit_mode, live_visualizer)
        loss_history.append(avg_loss)

        # Update live visualizer at end of epoch
        if live_visualizer is not None:
            live_visualizer.update_epoch_complete(avg_loss, model)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")
        print(f"Average loss: {avg_loss:.6f}")

        # Save checkpoint at intervals
        if not overfit_mode and (epoch + 1) % config["save_every"] == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                avg_loss,
                config["checkpoint_dir"],
                model_type="flow_matching",
                config=config,
            )
            print(f"Checkpoint saved at epoch {epoch + 1}")

        # Save best model
        if not overfit_mode and avg_loss < best_loss:
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

        # Visualize at intervals (only save static plots, live view is already updating)
        if (epoch + 1) % config.get("sample_every", 10) == 0 or epoch == start_epoch:
            print(f"Generating samples with {config['num_ode_steps']} ODE steps...")
            sample = visualize_training_progress(
                model=model,
                loss_history=loss_history,
                epoch=epoch + 1,
                config=config,
                overfit_mode=overfit_mode,
                original_image=original_image,
                save_dir=config["sample_dir"],
            )

            # Also save individual samples
            save_samples(sample[None, ...], epoch + 1, config["sample_dir"])

    return loss_history


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
        "--debug", action="store_true", help="Debug mode: use small dataset"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=250,
        help="Number of ODE steps for sampling (more steps = better quality, try 250-1000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (default: 32 for overfit, 128 for normal)",
    )
    parser.add_argument(
        "--conditional",
        action="store_true",
        help="Enable class-conditional generation (default: True)",
        default=True,
    )
    parser.add_argument(
        "--live_view",
        action="store_true",
        help="Enable live visualization window during training",
        default=False,
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
        "save_every": 100,
        "sample_every": 50 if args.overfit else 1,
        "num_ode_steps": args.num_steps,
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

    train_loader = CIFAR10DataLoader(
        train_images,
        train_labels,
        batch_size=config["batch_size"],
        shuffle=not args.overfit,  # Don't shuffle in overfit mode
    )

    print(f"Dataset: {len(train_images)} images")
    print(f"Batch size: {config['batch_size']}")
    print(f"Batches per epoch: {len(train_loader)}")

    use_conditioning = args.conditional

    print(f"\nInitializing Flow Matching model (conditional: {use_conditioning})...")
    model = FlowMatchModel(
        input_channels=3,
        hidden_channels=64,
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        time_emb_dim=256,
        num_classes=10 if use_conditioning else None,  # CIFAR-10 has 10 classes
        class_emb_dim=256 if use_conditioning else None,
    )

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
    print("Press Ctrl+C to save and exit gracefully")
    print("-" * 40)

    # Prepare original image for overfit mode visualization
    original_img_uint8 = None
    if args.overfit:
        # Get the original image for display
        original_img_uint8 = (single_image[0].transpose(1, 2, 0) * 255).astype(np.uint8)

    # Initialize live visualizer if requested
    live_visualizer = None
    if args.live_view:
        print("Initializing live visualization window...")
        try:
            live_visualizer = LiveTrainingVisualizer(
                config=config,
                overfit_mode=args.overfit,
                original_image=original_img_uint8
            )
            print("Live visualization window opened!")
        except Exception as e:
            print(f"Failed to initialize live visualization: {e}")
            print("Continuing without live visualization...")
            live_visualizer = None

    try:
        # Use the comprehensive training function
        loss_history_list = train_and_visualize(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            start_epoch=start_epoch,
            overfit_mode=args.overfit,
            original_image=original_img_uint8,
            live_visualizer=live_visualizer,
        )
    finally:
        # Clean up live visualizer
        if live_visualizer is not None:
            live_visualizer.close()

    # Update loss history for interrupt handler
    loss_history["epoch_losses"] = loss_history_list
    if loss_history_list:
        current_loss = loss_history_list[-1]

    # Final message
    if interrupt_handler.interrupted:
        print("\nTraining interrupted but progress saved!")
    else:
        print("\n" + "=" * 60)
        print("Training completed!")

    if loss_history_list:
        best_loss = min(loss_history_list)
        print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}")
    print(f"Samples saved to: {config['sample_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
