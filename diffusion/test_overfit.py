"""
Test script to verify DDPM training by overfitting on a single image
Includes live plot updates and sample generation
"""

import os
import sys
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image as PILImage

matplotlib.use("Agg")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.ddpm_model import DDPM_UNet
from diffusion.scheduler import NoiseScheduler
from data.cifar10 import load_train_data
from diffusion.ddpm_train import sample_images


def save_live_plot(losses, losses_by_timestep, timesteps_used, iteration):
    """Save current training plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss over iterations
    ax1.plot(losses, "b-", linewidth=0.5, alpha=0.7)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Training Loss (Iteration {iteration})")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    
    # Moving average of loss
    window = min(100, len(losses) // 2)
    if len(losses) >= window and window > 0:
        moving_avg = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax2.plot(range(window - 1, len(losses)), moving_avg, "g-", linewidth=2)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Loss (moving avg)")
        ax2.set_title(f"Moving Average (window={window})")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)
    
    # Loss by timestep (most recent value for each t)
    if losses_by_timestep:
        t_values = sorted(losses_by_timestep.keys())
        recent_losses = [losses_by_timestep[t][-1] for t in t_values]
        ax3.scatter(t_values, recent_losses, alpha=0.5, s=10)
        ax3.set_xlabel("Timestep")
        ax3.set_ylabel("Recent Loss")
        ax3.set_title(f"Loss by Timestep ({len(t_values)} unique timesteps)")
        ax3.set_yscale("log")
        ax3.grid(True, alpha=0.3)
    
    # Histogram of timesteps trained
    if timesteps_used:
        ax4.hist(timesteps_used[-1000:], bins=30, alpha=0.7, edgecolor='black')
        ax4.set_xlabel("Timestep")
        ax4.set_ylabel("Count (last 1000)")
        ax4.set_title("Recent Timestep Distribution")
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f"Training Progress - Iteration {iteration}", fontsize=16)
    plt.tight_layout()
    plt.savefig("./samples/overfit_test.png", dpi=100, bbox_inches="tight")
    plt.close()


def train_step(model, scheduler, optimizer, image, t_fixed=None):
    """Single DDPM training step on one image"""
    # Image comes in CHW format, convert to HWC for MLX
    image = mx.transpose(image, (0, 2, 3, 1))

    # Normalize to [-1, 1]
    image = image * 2.0 - 1.0

    batch_size = image.shape[0]

    # Use fixed timestep or sample random
    if t_fixed is not None:
        t = mx.array([t_fixed] * batch_size)
    else:
        t = mx.random.randint(0, scheduler.num_timesteps, (batch_size,))

    # Sample noise
    noise = mx.random.normal(image.shape)

    # Add noise to images (forward diffusion process)
    noisy_image = scheduler.add_noise(image, t, noise)

    @mx.compile
    def loss_fn(params):
        model.update(params)
        # Predict the noise
        predicted_noise = model(noisy_image, t)
        # L2 loss between predicted and actual noise
        loss = mx.mean((predicted_noise - noise) ** 2)
        return loss

    loss, grads = mx.value_and_grad(loss_fn)(model.parameters())
    optimizer.update(model, grads)
    mx.eval(loss)

    return loss, t[0].item()


def main():
    print("=" * 60)
    print("DDPM Overfit Test - Single Image")
    print("=" * 60)
    print("Training on single image with live plot updates")
    print("Check ./samples/overfit_test.png for progress")
    print("=" * 60)

    # Load one image from CIFAR-10
    print("\nLoading single CIFAR-10 image...")
    train_images, train_labels = load_train_data(
        data_dir="./cifar-10", download=True, normalize=True
    )

    # Take just one image and repeat it for batch
    single_image = mx.array(train_images[0:1])  # Shape: (1, 3, 32, 32)
    batch_size = 4
    fixed_batch = mx.repeat(single_image, batch_size, axis=0)  # Shape: (4, 3, 32, 32)

    print(f"Using single image repeated {batch_size} times")
    print(f"Image shape: {fixed_batch.shape}")
    
    # Save original image
    os.makedirs("./samples", exist_ok=True)
    original_np = np.array(mx.transpose(single_image, (0, 2, 3, 1))[0])
    original_uint8 = (original_np * 255).astype(np.uint8)
    PILImage.fromarray(original_uint8).save("./samples/overfit_original.png")
    print(f"Original image saved to ./samples/overfit_original.png")

    # Create model
    print("\nInitializing model...")
    model = DDPM_UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_multipliers=(1, 2, 2, 2),
        num_res_blocks=2,
        attention_resolutions=(8,),
    )

    # Create scheduler
    scheduler = NoiseScheduler(
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
    )

    # Create optimizer with higher learning rate for faster convergence
    optimizer = optim.Adam(learning_rate=1e-3)

    # Training loop
    num_iterations = 10000
    plot_every = 100  # Update plot every N iterations
    sample_every = 2000  # Generate samples every N iterations
    
    print(f"\nTraining for {num_iterations} iterations...")
    print(f"Plot updates every {plot_every} iterations")
    print(f"Sample generation every {sample_every} iterations")
    print("-" * 40)

    losses = []
    timesteps_used = []
    losses_by_timestep = {}

    for i in range(num_iterations):
        # Training step
        loss, t_used = train_step(model, scheduler, optimizer, fixed_batch)
        loss_val = loss.item()
        losses.append(loss_val)
        timesteps_used.append(t_used)
        
        # Track loss by timestep
        if t_used not in losses_by_timestep:
            losses_by_timestep[t_used] = []
        losses_by_timestep[t_used].append(loss_val)
        
        # Print progress
        if i % 100 == 0:
            recent_avg = np.mean(losses[-100:]) if len(losses) >= 100 else loss_val
            print(f"Iter {i:5d}: Loss = {loss_val:.6f}, Avg = {recent_avg:.6f}, t = {t_used:3d}, Timesteps seen = {len(losses_by_timestep)}")
        
        # Update plot
        if (i + 1) % plot_every == 0:
            save_live_plot(losses, losses_by_timestep, timesteps_used, i + 1)
            
            # Show statistics
            if (i + 1) % 1000 == 0:
                print(f"\n  === Statistics at iteration {i+1} ===")
                print(f"  Timestep coverage: {len(losses_by_timestep)}/{scheduler.num_timesteps}")
                print(f"  Average loss (last 500): {np.mean(losses[-500:]):.6f}")
                print(f"  Min loss achieved: {min(losses):.6f}")
                
                # Show loss by timestep ranges
                ranges = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]
                print("  Loss by timestep range:")
                for t_min, t_max in ranges:
                    range_losses = [losses_by_timestep[t][-1] for t in losses_by_timestep 
                                  if t_min <= t < t_max and losses_by_timestep[t]]
                    if range_losses:
                        avg = np.mean(range_losses)
                        print(f"    t=[{t_min:3d},{t_max:3d}): avg = {avg:.6f}")
                print()
        
        # Generate samples periodically
        if (i + 1) % sample_every == 0:
            print(f"\n  Generating samples at iteration {i+1}...")
            samples = sample_images(model, scheduler, num_samples=4)
            
            # Save samples
            samples_np = np.array(samples)
            samples_uint8 = (samples_np * 255).astype(np.uint8)
            
            # Create 2x2 grid
            grid_img = np.zeros((64, 64, 3), dtype=np.uint8)
            for idx, sample in enumerate(samples_uint8[:4]):
                row = idx // 2
                col = idx % 2
                grid_img[row*32:(row+1)*32, col*32:(col+1)*32] = sample
            
            PILImage.fromarray(grid_img).save(f"./samples/overfit_samples_iter_{i+1}.png")
            print(f"  Samples saved to ./samples/overfit_samples_iter_{i+1}.png\n")

    # Final analysis
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Loss reduction: {(1 - losses[-1]/losses[0]) * 100:.1f}%")
    print(f"Minimum loss achieved: {min(losses):.6f}")
    print(f"Timestep coverage: {len(losses_by_timestep)}/{scheduler.num_timesteps} timesteps")
    
    # Final samples
    print("\nGenerating final samples...")
    samples = sample_images(model, scheduler, num_samples=16)
    samples_np = np.array(samples)
    samples_uint8 = (samples_np * 255).astype(np.uint8)
    
    # Create 4x4 grid for final samples
    grid_img = np.zeros((128, 128, 3), dtype=np.uint8)
    for idx, sample in enumerate(samples_uint8[:16]):
        row = idx // 4
        col = idx % 4
        if row < 4 and col < 4:
            grid_img[row*32:(row+1)*32, col*32:(col+1)*32] = sample
    
    PILImage.fromarray(grid_img).save("./samples/overfit_final_samples.png")
    print("Final samples saved to ./samples/overfit_final_samples.png")
    
    # Save final plot
    save_live_plot(losses, losses_by_timestep, timesteps_used, num_iterations)
    print("Final plot saved to ./samples/overfit_test.png")

    # Check if model is learning
    if losses[-1] < losses[0] * 0.01:  # Loss dropped by 99%
        print("\n✓ EXCELLENT: Model is learning! Loss decreased by >99%.")
    elif losses[-1] < losses[0] * 0.1:  # Loss dropped by 90%
        print("\n✓ SUCCESS: Model is learning well! Loss decreased by >90%.")
    elif losses[-1] < losses[0] * 0.5:  # Loss dropped by 50%
        print("\n⚠ PARTIAL SUCCESS: Model is learning but slowly.")
    else:
        print("\n✗ FAILURE: Model is not learning effectively.")
        print("  Check model architecture and training loop.")

    print("=" * 60)


if __name__ == "__main__":
    main()