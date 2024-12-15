import os
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from model import YOLO
from loss import yolo_loss
from data.voc import VOCDataset, create_data_loader
import json


def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    model_state = model.parameters()
    mx.savez(os.path.join(save_dir, f"yolo_epoch_{epoch}.npz"), **model_state)

    # Save optimizer state
    optimizer_state = optimizer.state
    mx.savez(os.path.join(save_dir, f"optimizer_epoch_{epoch}.npz"), **optimizer_state)

    # Save training info
    info = {
        "epoch": epoch,
        "loss": float(loss.item()),
    }
    with open(os.path.join(save_dir, f"info_epoch_{epoch}.json"), "w") as f:
        json.dump(info, f)


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
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0005,
    momentum: float = 0.9,
    resume_epoch: int = None,
):
    """Train YOLO model"""
    # Create model and optimizer
    model = YOLO()
    optimizer = optim.Adam(learning_rate=learning_rate, weight_decay=weight_decay)

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_epoch is not None:
        start_epoch, _ = load_checkpoint(model, optimizer, save_dir, resume_epoch)
        print(f"Resumed from epoch {start_epoch}")

    # Create dataset and data loader
    dataset = VOCDataset(data_dir, year="2012", image_set="train")
    val_dataset = VOCDataset(data_dir, year="2012", image_set="val")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train = True  # Set to training mode
        epoch_loss = 0.0
        start_time = time.time()

        # Get training batches
        train_images, train_targets = create_data_loader(
            dataset, batch_size, shuffle=True
        )

        # Train for one epoch
        for batch_idx, (images, targets) in enumerate(zip(train_images, train_targets)):

            def loss_fn(model):
                predictions = model(images)
                return yolo_loss(predictions, targets)

            # Compute loss and gradients
            loss, grads = mx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)
            epoch_loss += loss.item()

            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Batch [{batch_idx+1}/{len(train_images)}], "
                    f"Loss: {loss.item():.4f}"
                )

        # Compute epoch statistics
        avg_loss = epoch_loss / len(train_images)
        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Average Loss: {avg_loss:.4f}, "
            f"Time: {epoch_time:.2f}s"
        )

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, save_dir)
            print(f"Checkpoint saved at epoch {epoch+1}")


if __name__ == "__main__":
    # Training configuration
    config = {
        "data_dir": "./VOCdevkit/VOC2012",  # Path to VOC dataset
        "save_dir": "./checkpoints",  # Where to save model checkpoints
        "num_epochs": 135,  # Total epochs to train
        "batch_size": 64,  # Batch size
        "learning_rate": 0.001,  # Initial learning rate
        "weight_decay": 0.0005,  # Weight decay for regularization
        "momentum": 0.9,  # SGD momentum
        "resume_epoch": None,  # Set to epoch number to resume training
    }

    train(**config)
