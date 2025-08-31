"""
Test script to verify DDPM training by overfitting on a single image
This should drive the loss to near zero if everything is working correctly
"""

import os
import sys
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.ddpm_model import DDPM_UNet
from diffusion.scheduler import NoiseScheduler
from data.cifar10 import load_train_data


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
    print("This test trains on a single image repeatedly")
    print("Loss should drop to near zero if training works correctly")
    print("=" * 60)
    
    # Load one image from CIFAR-10
    print("\nLoading single CIFAR-10 image...")
    train_images, train_labels = load_train_data(
        data_dir="./cifar-10", 
        download=True, 
        normalize=True
    )
    
    # Take just one image and repeat it for batch
    single_image = mx.array(train_images[0:1])  # Shape: (1, 3, 32, 32)
    batch_size = 4
    fixed_batch = mx.repeat(single_image, batch_size, axis=0)  # Shape: (4, 3, 32, 32)
    
    print(f"Using single image repeated {batch_size} times")
    print(f"Image shape: {fixed_batch.shape}")
    
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
    num_iterations = 1000
    print(f"\nTraining for {num_iterations} iterations on same image...")
    print("-" * 40)
    
    losses = []
    timesteps_used = []
    
    # Test 1: Train with random timesteps
    print("\nPhase 1: Training with random timesteps")
    for i in range(num_iterations // 2):
        loss, t_used = train_step(model, scheduler, optimizer, fixed_batch)
        losses.append(loss.item())
        timesteps_used.append(t_used)
        
        if i % 50 == 0:
            print(f"Iteration {i:4d}: Loss = {loss.item():.6f}, t = {t_used}")
    
    # Test 2: Train with fixed timestep
    print("\nPhase 2: Training with fixed timestep (t=500)")
    fixed_t = 500
    for i in range(num_iterations // 2, num_iterations):
        loss, t_used = train_step(model, scheduler, optimizer, fixed_batch, t_fixed=fixed_t)
        losses.append(loss.item())
        timesteps_used.append(t_used)
        
        if i % 50 == 0:
            print(f"Iteration {i:4d}: Loss = {loss.item():.6f}, t = {t_used}")
    
    # Analyze results
    print("\n" + "=" * 60)
    print("Results Analysis:")
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Loss reduction: {(1 - losses[-1]/losses[0]) * 100:.1f}%")
    print(f"Minimum loss achieved: {min(losses):.6f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss over iterations
    ax1.plot(losses, 'b-', linewidth=1)
    ax1.axvline(x=num_iterations//2, color='r', linestyle='--', 
                label='Switch to fixed t')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss on Single Image')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Moving average of loss
    window = 50
    if len(losses) >= window:
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(losses)), moving_avg, 'g-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss (moving avg)')
        ax2.set_title(f'Moving Average (window={window})')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./samples/overfit_test.png', dpi=100, bbox_inches='tight')
    print(f"\nPlot saved to ./samples/overfit_test.png")
    
    # Check if model is learning
    if losses[-1] < losses[0] * 0.1:  # Loss dropped by 90%
        print("\n✓ SUCCESS: Model is learning! Loss decreased significantly.")
    elif losses[-1] < losses[0] * 0.5:  # Loss dropped by 50%
        print("\n⚠ PARTIAL SUCCESS: Model is learning but slowly.")
    else:
        print("\n✗ FAILURE: Model is not learning effectively.")
        print("  Check model architecture and training loop.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()