import os
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from model import UNet
from scheduler import NoiseScheduler
from loss import diffusion_loss
from data.voc import VOCDiffusionDataset, create_data_loader
import json


def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    try:
        save_path = os.path.join(save_dir, f"diffusion_epoch_{epoch}.npz")
        model.save_weights(save_path)
        print(f"Successfully saved model to {save_path}")
    except Exception as e:
        print(f"Error saving model state: {str(e)}")
        raise

    # Save optimizer state
    try:
        save_path = os.path.join(save_dir, f"optimizer_epoch_{epoch}.npz")
        save_dict = {
            "learning_rate": mx.array(optimizer.learning_rate, dtype=mx.float32),
            "step": mx.array(optimizer.state.get("step", 0), dtype=mx.int32),
        }
        mx.savez(save_path, **save_dict)
        print(f"Successfully saved optimizer state to {save_path}")
    except Exception as e:
        print(f"Error saving optimizer state: {str(e)}")
        raise

    # Save training info
    info = {
        "epoch": epoch,
        "loss": loss,
    }
    info_path = os.path.join(save_dir, f"info_epoch_{epoch}.json")
    with open(info_path, "w") as f:
        json.dump(info, f)
    print(f"Successfully saved training info to {info_path}")


def load_checkpoint(model, optimizer, checkpoint_dir, epoch):
    """Load model checkpoint"""
    # Load model weights
    model_path = os.path.join(checkpoint_dir, f"diffusion_epoch_{epoch}.npz")
    if os.path.exists(model_path):
        model.load_weights(model_path)
        print(f"Successfully loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"No model checkpoint found at {model_path}")

    # Load optimizer state
    optim_path = os.path.join(checkpoint_dir, f"optimizer_epoch_{epoch}.npz")
    if os.path.exists(optim_path):
        optim_state = mx.load(optim_path)
        optimizer.learning_rate = float(optim_state["learning_rate"])
        optimizer.state["step"] = int(optim_state["step"])
        print(f"Successfully loaded optimizer state from {optim_path}")
    else:
        print(f"No optimizer state found at {optim_path}, using default values")


def train_step(model, scheduler, optimizer, images, t):
    """Single training step"""
    noise = mx.random.normal(images.shape)
    noisy_images = scheduler.q_sample(images, t, noise=noise)
    
    def loss_fn(model_params):
        predicted_noise = model.apply(model_params, noisy_images, t)
        return diffusion_loss(predicted_noise, noise)
    
    loss, grads = nn.value_and_grad(model, loss_fn)(model.parameters())
    optimizer.update(model, grads)
    mx.eval(model.parameters())
    
    return loss


def train(
    data_dir: str,
    save_dir: str,
    image_size: int = 64,
    batch_size: int = 8,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    resume_epoch: int | None = None,
):
    """Train diffusion model"""
    print("Initializing model and optimizer...")
    model = UNet(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_levels=[2],
        channel_mult=(1, 2, 4, 8),
        time_emb_dim=128
    )
    
    optimizer = optim.Adam(learning_rate=learning_rate)
    scheduler = NoiseScheduler()

    # Initialize dataset and dataloader
    print("Setting up dataset...")
    dataset = VOCDiffusionDataset(data_dir=data_dir, img_size=image_size)
    dataloader = create_data_loader(dataset, batch_size=batch_size)
    print(f"Dataset size: {len(dataset)} images")

    # Resume from checkpoint if specified
    if resume_epoch is not None:
        load_checkpoint(model, optimizer, save_dir, resume_epoch)
        start_epoch = resume_epoch + 1
    else:
        start_epoch = 0

    print("Starting training...")
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        start_time = time.time()

        for batch_idx, (images, descriptions) in enumerate(dataloader):
            # Sample random timesteps
            t = mx.random.randint(0, scheduler.num_timesteps, (images.shape[0],))
            
            # Training step
            loss = train_step(model, scheduler, optimizer, images, t)
            epoch_loss += loss

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss:.4f}")
                print(f"Sample descriptions:")
                for desc in descriptions[:2]:  # Print first two descriptions
                    print(f"  - {desc}")

        epoch_loss /= len(dataloader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}: Average Loss = {epoch_loss:.4f}, Time = {epoch_time:.2f}s")

        # Save checkpoint
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch, float(epoch_loss), save_dir)

        # Generate and save sample images
        if epoch % 10 == 0:
            print("Generating samples...")
            samples = scheduler.sample(model, image_size=image_size, batch_size=4)
            # TODO: Add image saving logic here

    print("Training completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Diffusion Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to VOC dataset directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--image_size", type=int, default=64, help="Size of images")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume_epoch", type=int, help="Resume from epoch")

    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        resume_epoch=args.resume_epoch,
    )
