import os
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from model import YOLO
from loss import yolo_loss
from data.voc import VOCDataset, create_data_loader
import json
from mlx.utils import tree_flatten, tree_map


def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    try:
        save_path = os.path.join(save_dir, f"yolo_epoch_{epoch}.npz")
        flat_params = tree_flatten(model.parameters())
        mx.savez(save_path, **dict(flat_params))
        print(f"Successfully saved model to {save_path}")
    except Exception as e:
        print(f"Error saving model state: {str(e)}")
        raise

    # Save optimizer state
    try:
        save_path = os.path.join(save_dir, f"optimizer_epoch_{epoch}.npz")
        flat_opt_state = tree_flatten(optimizer.state)
        mx.savez(save_path, **dict(flat_opt_state))
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
    model_path = os.path.join(checkpoint_dir, f"yolo_epoch_{epoch}.npz")
    model_state = mx.load(model_path)
    model.update(model_state)

    # Load optimizer state
    optimizer_path = os.path.join(checkpoint_dir, f"optimizer_epoch_{epoch}.npz")
    optimizer_state = mx.load(optimizer_path)
    optimizer.state.update(optimizer_state)

    # Load training info
    info_path = os.path.join(checkpoint_dir, f"info_epoch_{epoch}.json")
    with open(info_path, "r") as f:
        info = json.load(f)

    return info["epoch"], info["loss"]


def train(
    data_dir: str,
    save_dir: str,
    num_epochs: int = 135,
    batch_size: int = 8,
    accumulation_steps: int = 2,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    resume_epoch: int | None = None,
):
    """Train YOLO model"""
    # Create model and optimizer
    print("Creating model...")
    model = YOLO()
    optimizer = optim.Adam(
        learning_rate=learning_rate, betas=[beta1, beta2], eps=epsilon
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_epoch is not None:
        start_epoch, _ = load_checkpoint(model, optimizer, save_dir, resume_epoch)
        print(f"Resumed from epoch {start_epoch}")

    # Create dataset and data loader
    print("Loading dataset...")
    dataset = VOCDataset(data_dir, year="2012", image_set="train")
    val_dataset = VOCDataset(data_dir, year="2012", image_set="val")

    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, num_epochs):
        model.train(True)  # Set to training mode
        epoch_loss = 0.0
        start_time = time.time()

        # Get training batches
        train_images, train_targets = create_data_loader(
            dataset, batch_size, shuffle=True
        )

        accumulated_grads = None
        # Train for one epoch
        for batch_idx, (images, targets) in enumerate(zip(train_images, train_targets)):

            def loss_fn(params):
                model.update(params)
                predictions = model(images)
                return yolo_loss(predictions, targets)

            # Compute loss and gradients
            loss, grads = mx.value_and_grad(loss_fn)(model.parameters())

            # Scale gradients for accumulation
            grads = tree_map(lambda x: x / accumulation_steps, grads)

            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(
                    lambda x, y: x + y, accumulated_grads, grads
                )

            # Update weights after accumulation steps
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.update(model, accumulated_grads)
                accumulated_grads = None
                mx.eval(model.parameters())  # Force evaluation to free memory

            epoch_loss += loss.item()

            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Batch [{batch_idx+1}/{len(train_images)}], "
                    f"Loss: {loss.item():.4f}"
                )

            # Force memory cleanup
            if (batch_idx + 1) % 5 == 0:
                mx.eval(model.parameters())

        # Update with any remaining accumulated gradients
        if accumulated_grads is not None:
            optimizer.update(model, accumulated_grads)
            mx.eval(model.parameters())

        # Compute epoch statistics
        avg_loss = epoch_loss / len(train_images)
        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Average Loss: {avg_loss:.4f}, "
            f"Time: {epoch_time:.2f}s"
        )

        # Save checkpoint every 2 epochs
        if epoch % 2 == 0 or epoch == num_epochs - 1:
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, save_dir)
            print(f"Checkpoint saved at epoch {epoch+1}")


if __name__ == "__main__":
    # Training configuration
    config = {
        "data_dir": "./VOCdevkit/VOC2012",
        "save_dir": "./checkpoints",
        "num_epochs": 135,
        "batch_size": 32,
        "accumulation_steps": 2,
        "learning_rate": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "resume_epoch": None,
    }

    train(**config)
