"""
Conditional DDPM Training Script for CIFAR-10
Trains a class-conditional diffusion model using CLIP-style embeddings

This script implements:
- Class-conditional training with dropout for classifier-free guidance
- Conditional sampling with guidance scale
- Auto-resume from checkpoints
- Graceful interruption handling
"""

import os
import sys
import time
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.ddpm_conditional_model import ConditionalDDPM_UNet
from diffusion.scheduler import NoiseScheduler
from diffusion.utils import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint_in_dir,
    create_interrupt_handler,
)
from data.cifar10 import load_train_data, CIFAR10DataLoader

# CIFAR-10 class names for reference
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


def train_step(model, scheduler, optimizer, images, labels, unconditional_prob=0.1):
    """
    Single conditional DDPM training step

    Args:
        model: Conditional DDPM model
        scheduler: Noise scheduler
        optimizer: Optimizer
        images: Batch of images [batch, channels, height, width]
        labels: Class labels [batch]
        unconditional_prob: Probability of dropping class info for CFG training
    """
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
        # Predict the noise with class conditioning
        predicted_noise = model(noisy_images, t, labels, unconditional_prob)
        # L2 loss between predicted and actual noise
        loss = mx.mean((predicted_noise - noise) ** 2)
        return loss

    # Compile the loss function for better performance
    loss_fn = mx.compile(loss_fn)

    loss, grads = mx.value_and_grad(loss_fn)(model.parameters())
    optimizer.update(model, grads)
    mx.eval(loss)

    return loss


def sample_images_conditional(
    model, scheduler, class_labels=None, num_samples=4, guidance_scale=7.5, seed=None
):
    """
    Generate images with class conditioning using classifier-free guidance

    Args:
        model: Conditional DDPM model
        scheduler: Noise scheduler
        class_labels: List/array of class indices to generate (None for random)
        num_samples: Number of samples to generate
        guidance_scale: Strength of class conditioning (1.0 = no guidance, >1 = stronger)
        seed: Random seed for reproducibility

    Returns:
        Generated images [num_samples, height, width, channels]
    """
    if seed is not None:
        mx.random.seed(seed)

    # Set up class labels
    if class_labels is None:
        # Generate random classes
        class_labels = mx.random.randint(0, 10, (num_samples,))
    elif isinstance(class_labels, int):
        # Single class for all samples
        class_labels = mx.array([class_labels] * num_samples)
    else:
        # Convert list to array
        class_labels = mx.array(class_labels)

    # Start from pure noise (CHW format for scheduler)
    shape = (num_samples, 3, 32, 32)
    img = mx.random.normal(shape)

    # Reverse diffusion process
    for i in reversed(range(scheduler.num_timesteps)):
        t = mx.array([i] * num_samples)

        # Convert to HWC for model
        img_hwc = mx.transpose(img, (0, 2, 3, 1))

        # Get predictions with and without class conditioning (for CFG)
        if guidance_scale > 1.0:
            # Conditional prediction
            noise_pred_cond = model(img_hwc, t, class_labels, unconditional_prob=0.0)

            # Unconditional prediction (using unconditional token)
            uncond_labels = mx.ones_like(class_labels) * model.num_classes
            noise_pred_uncond = model(img_hwc, t, uncond_labels, unconditional_prob=0.0)

            # Classifier-free guidance
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            # Just use conditional prediction
            noise_pred = model(img_hwc, t, class_labels, unconditional_prob=0.0)

        # Convert back to CHW for scheduler
        noise_pred = mx.transpose(noise_pred, (0, 3, 1, 2))

        # Denoise step
        if i > 0:
            # Get noise variance for this timestep
            noise = mx.random.normal(img.shape)
            beta_t = scheduler.betas[i]
            sqrt_one_minus_alpha_cumprod_t = scheduler.sqrt_one_minus_alphas_cumprod[i]
            sqrt_recip_alphas_t = scheduler.sqrt_recip_alphas[i]

            # Update image
            img = sqrt_recip_alphas_t * (
                img - beta_t * noise_pred / sqrt_one_minus_alpha_cumprod_t
            )

            # Add noise (except for last step)
            posterior_variance_t = scheduler.posterior_variance[i]
            img = img + mx.sqrt(posterior_variance_t) * noise
        else:
            # Final step without noise
            beta_t = scheduler.betas[i]
            sqrt_one_minus_alpha_cumprod_t = scheduler.sqrt_one_minus_alphas_cumprod[i]
            sqrt_recip_alphas_t = scheduler.sqrt_recip_alphas[i]

            img = sqrt_recip_alphas_t * (
                img - beta_t * noise_pred / sqrt_one_minus_alpha_cumprod_t
            )

    # Convert to HWC and denormalize
    img = mx.transpose(img, (0, 2, 3, 1))
    img = (img + 1.0) / 2.0
    img = mx.clip(img, 0.0, 1.0)

    return img, class_labels


def save_conditional_samples(samples, labels, epoch, save_dir="./samples"):
    """
    Save generated samples with their class labels

    Args:
        samples: Generated images
        labels: Class labels for each sample
        epoch: Current epoch number
        save_dir: Directory to save samples
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert to numpy
    samples_np = np.array(samples)
    labels_np = np.array(labels)

    # Convert to uint8
    samples_uint8 = (samples_np * 255).astype(np.uint8)

    # Create grid with labels
    num_samples = len(samples_uint8)
    grid_size = int(np.sqrt(num_samples))
    if grid_size * grid_size < num_samples:
        grid_size += 1

    # Create grid image with space for labels
    img_size = 32
    padding = 5
    label_height = 20
    cell_size = img_size + padding * 2 + label_height

    grid_height = grid_size * cell_size
    grid_width = grid_size * cell_size

    # Create white background
    grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

    # Use PIL to add text labels
    pil_img = Image.fromarray(grid_img)
    draw = ImageDraw.Draw(pil_img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font = ImageFont.load_default()

    # Place images in grid with labels
    for idx, (sample, label) in enumerate(zip(samples_uint8, labels_np)):
        if idx >= grid_size * grid_size:
            break
        row = idx // grid_size
        col = idx % grid_size

        y_start = row * cell_size + padding
        x_start = col * cell_size + padding

        # Place the image
        pil_img.paste(Image.fromarray(sample), (x_start, y_start))
        
        # Add label text below the image
        label_text = CIFAR10_CLASSES[label]
        text_y = y_start + img_size + 2
        text_x = x_start + img_size // 2
        
        # Draw text centered below image
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text((text_x - text_width // 2, text_y), label_text, fill=(0, 0, 0), font=font)

    # Save grid
    grid_path = os.path.join(save_dir, f"conditional_samples_epoch_{epoch}.png")
    pil_img.save(grid_path)

    # Print info
    print(f"  Generated {num_samples} conditional samples")
    print(
        f"  Classes: {[CIFAR10_CLASSES[l] for l in labels_np[:min(10, len(labels_np))]]}"
    )
    print(f"  Saved to {grid_path}")


def train_epoch(
    model,
    scheduler,
    optimizer,
    train_loader,
    epoch,
    unconditional_prob=0.1,
    sample_every=500,
    loss_history=None,
):
    """
    Train for one epoch with conditional generation

    Args:
        model: Conditional DDPM model
        scheduler: Noise scheduler
        optimizer: Optimizer
        train_loader: DataLoader
        epoch: Current epoch number
        unconditional_prob: Dropout probability for CFG training
        sample_every: Generate samples every N batches
        loss_history: Dictionary to track losses
    """
    total_loss = 0
    num_batches = 0
    batch_start_time = time.time()
    batch_losses = []

    for batch_idx, (images, labels) in enumerate(train_loader):
        # Training step with class labels
        loss = train_step(
            model, scheduler, optimizer, images, labels, unconditional_prob
        )
        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1
        batch_losses.append(loss_val)

        # Print progress
        if batch_idx % 10 == 0:
            batch_time = time.time() - batch_start_time
            avg_loss = total_loss / num_batches
            print(
                f"  Batch {batch_idx}/{len(train_loader)}: "
                f"Loss = {loss_val:.4f}, Avg = {avg_loss:.4f}, "
                f"Time = {batch_time:.2f}s"
            )
            batch_start_time = time.time()

            # Record loss
            if loss_history is not None:
                loss_history["batch_losses"].append(
                    {
                        "epoch": epoch,
                        "batch": batch_idx,
                        "loss": loss_val,
                        "avg_loss": avg_loss,
                    }
                )

        # Generate conditional samples periodically (skip first batch)
        if batch_idx % sample_every == 0 and batch_idx > 0:
            print(f"  Generating conditional samples at batch {batch_idx}...")

            # Generate samples for each class
            samples_list = []
            labels_list = []

            for class_idx in range(2):
                class_samples, class_labels = sample_images_conditional(
                    model,
                    scheduler,
                    class_labels=class_idx,
                    num_samples=1,
                    guidance_scale=7.5,
                )
                samples_list.append(class_samples)
                labels_list.append(class_labels)

            # Combine all samples
            all_samples = mx.concatenate(samples_list, axis=0)
            all_labels = mx.concatenate(labels_list, axis=0)

            # Save samples
            save_conditional_samples(all_samples, all_labels, epoch * 1000 + batch_idx)

    return total_loss / num_batches, batch_losses


def main():
    print("=" * 60)
    print("Conditional DDPM Training for CIFAR-10")
    print("with CLIP-style Class Embeddings")
    print("=" * 60)

    # Configuration
    config = {
        "data_dir": "./cifar-10",
        "checkpoint_dir": "./checkpoints/ddpm_conditional",
        "sample_dir": "./samples/ddpm_conditional",
        "batch_size": 32,
        "learning_rate": 2e-4,
        "num_epochs": 100,
        "num_timesteps": 1000,
        "unconditional_prob": 0.1,  # 10% dropout for CFG training
        "guidance_scale": 7.5,  # For sampling
        "sample_every": 50,
        "save_every": 5,
        "resume_from": None,
    }

    # Load data
    print("\nLoading CIFAR-10 with labels...")
    train_images, train_labels = load_train_data(
        data_dir=config["data_dir"], download=True, normalize=True
    )

    # Debug mode
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
    print(f"Classes: {len(np.unique(train_labels))} ({', '.join(CIFAR10_CLASSES)})")
    print(f"Batch size: {config['batch_size']}")
    print(f"Batches per epoch: {len(train_loader)}")

    # Create conditional model
    print("\nInitializing conditional model...")
    model = ConditionalDDPM_UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_multipliers=(1, 2, 2, 2),
        num_res_blocks=2,
        attention_resolutions=(8,),
        num_classes=10,  # CIFAR-10 has 10 classes
        class_emb_dim=128,
        use_cross_attention=True,  # Enable cross-attention to class embeddings
    )

    # Create scheduler
    scheduler = NoiseScheduler(
        num_timesteps=config["num_timesteps"],
        beta_start=1e-4,
        beta_end=0.02,
    )

    # Create optimizer
    optimizer = optim.Adam(learning_rate=config["learning_rate"])

    # Auto-load latest checkpoint
    start_epoch = 0
    checkpoint_to_load = config["resume_from"]

    if not checkpoint_to_load:
        latest_checkpoint = find_latest_checkpoint_in_dir(config["checkpoint_dir"])
        if latest_checkpoint:
            print(f"\nFound existing checkpoint: {latest_checkpoint}")
            checkpoint_to_load = latest_checkpoint

    if checkpoint_to_load:
        print(f"\nLoading checkpoint from {checkpoint_to_load}...")
        try:
            start_epoch = load_checkpoint(
                model, optimizer, checkpoint_to_load, expected_type="conditional"
            )
            print(f"Resumed conditional model from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting from scratch...")

    # Set up interrupt handler for graceful shutdown
    current_epoch = start_epoch
    current_loss = float("inf")

    # Create interrupt handler with save callback
    interrupt_handler = create_interrupt_handler(
        model,
        optimizer,
        epoch_getter=lambda: current_epoch + 1,
        loss_getter=lambda: (
            loss_history["epoch_losses"][-1]
            if loss_history["epoch_losses"]
            else current_loss
        ),
        checkpoint_dir=config["checkpoint_dir"],
        model_type="conditional",
        config=config,
    )
    interrupt_handler.setup()

    # Training loop
    print(f"\nStarting conditional training from epoch {start_epoch}...")
    print("Press Ctrl+C to save and exit gracefully")
    print("-" * 40)

    best_loss = float("inf")
    loss_history = {"epoch_losses": [], "batch_losses": [], "config": config}

    for epoch in range(start_epoch, config["num_epochs"]):
        current_epoch = epoch
        if interrupt_handler.should_stop:
            break

        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        start_time = time.time()

        # Train one epoch
        avg_loss, batch_losses = train_epoch(
            model,
            scheduler,
            optimizer,
            train_loader,
            epoch,
            unconditional_prob=config["unconditional_prob"],
            sample_every=config["sample_every"],
            loss_history=loss_history,
        )

        # Track epoch loss
        loss_history["epoch_losses"].append(avg_loss)
        current_loss = avg_loss

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")
        print(f"Average loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % config["save_every"] == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                avg_loss,
                config["checkpoint_dir"],
                model_type="conditional",
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
                model_type="conditional",
                config=config,
            )
            print(f"New best conditional model saved (loss: {best_loss:.4f})")

        # Generate samples for all classes
        print(f"\nGenerating conditional samples for epoch {epoch + 1}...")
        samples_list = []
        labels_list = []

        for class_idx in range(10):
            samples, labels = sample_images_conditional(
                model,
                scheduler,
                class_labels=class_idx,
                num_samples=2,
                guidance_scale=config["guidance_scale"],
            )
            samples_list.append(samples)
            labels_list.append(labels)

        all_samples = mx.concatenate(samples_list, axis=0)
        all_labels = mx.concatenate(labels_list, axis=0)
        save_conditional_samples(
            all_samples, all_labels, epoch + 1, config["sample_dir"]
        )

        if interrupt_handler.should_stop:
            break

    # Final message
    if interrupt_handler.interrupted:
        print("\nTraining interrupted but progress saved!")

    else:
        print("\n" + "=" * 60)
        print("Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}")
    print(f"Samples saved to: {config['sample_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
