import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from model import YOLO
from data.voc import VOCDataset, create_data_loader
import time
from tabulate import tabulate
from functools import partial
import argparse
from pathlib import Path


def bbox_loss(predictions, targets, model):
    """
    Simplified loss focusing only on bounding box coordinate regression.
    Only computes coordinate loss (x, y, w, h) without confidence scores.
    """
    batch_size = predictions.shape[0]
    S = model.S  # Grid size
    B = model.B  # Number of boxes per cell

    # Reshape predictions and targets
    pred = mx.reshape(predictions[..., : B * 4], (batch_size, S, S, B, 4))
    targ = mx.reshape(targets[..., :4], (batch_size, S, S, 1, 4))
    targ = mx.repeat(targ, B, axis=3)  # Repeat for each predicted box

    # Extract coordinates
    pred_xy = pred[..., :2]  # Center coordinates
    pred_wh = pred[..., 2:4]  # Width and height
    targ_xy = targ[..., :2]
    targ_wh = targ[..., 2:4]

    # Get object mask from the original targets
    obj_mask = mx.squeeze(
        mx.reshape(targets[..., 4:5], (batch_size, S, S, 1)), axis=-1
    )  # Shape: [batch, S, S]
    obj_mask = mx.repeat(
        mx.expand_dims(obj_mask, axis=-1), B, axis=-1
    )  # Shape: [batch, S, S, B]

    # Position loss (normalized by cell size)
    xy_loss = mx.sum(mx.square(pred_xy - targ_xy), axis=-1) / (S * S)

    # Size loss (using relative scale)
    wh_loss = mx.sum(
        mx.square(
            mx.sqrt(mx.maximum(pred_wh, 1e-6)) - mx.sqrt(mx.maximum(targ_wh, 1e-6))
        ),
        axis=-1,
    )

    # Only compute loss for cells that contain objects
    coord_loss = obj_mask * (xy_loss + 0.5 * wh_loss)  # Reduce weight of size loss

    # Normalize by number of objects and add small lambda
    num_objects = mx.sum(obj_mask) + 1e-6
    total_loss = 2.0 * mx.sum(coord_loss) / num_objects  # Reduced coordinate weight

    # Return raw values for monitoring
    xy_total = mx.sum(obj_mask * xy_loss)
    wh_total = mx.sum(obj_mask * wh_loss)

    return total_loss, (xy_total, wh_total, num_objects)


def train_step(model, batch, optimizer):
    """Single training step with coordinate-only loss"""
    images, targets = batch

    def loss_fn(params, images, targets):
        model.update(params)
        predictions = model(images)
        return bbox_loss(predictions, targets, model)

    # Compute loss and gradients
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Capture state for compilation
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def compiled_step():
        (loss, components), grads = loss_and_grad_fn(
            model.parameters(), images, targets
        )
        optimizer.update(model, grads)
        return loss, components

    loss, (xy_total, wh_total, num_objects) = compiled_step()

    # Calculate components after compilation
    components = {
        "xy": xy_total.item() / num_objects.item(),
        "wh": wh_total.item() / num_objects.item(),
    }

    return loss, components


def validate(model, val_loader):
    """Run validation lazily"""
    model.eval()
    losses = []

    for batch in val_loader:
        images, targets = batch
        predictions = model(images)
        loss, _ = bbox_loss(predictions, targets, model)
        losses.append(loss)

    # Only evaluate at the end
    mx.eval(losses)
    return sum(l.item() for l in losses) / len(losses)


def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights using safetensors
    model_path = os.path.join(save_dir, f"model_epoch_{epoch}.safetensors")
    print(f"\nSaving model to {model_path}")
    print("Model parameters before saving:")
    for name, param in model.parameters().items():
        print(f"  {name}: {param.shape if hasattr(param, 'shape') else 'no shape'}")

    model.save_weights(model_path)

    # Verify saved weights
    print("\nVerifying saved weights:")
    temp_model = YOLO()
    temp_model.load_weights(model_path)
    for name, param in temp_model.parameters().items():
        print(f"  {name}: {param.shape if hasattr(param, 'shape') else 'no shape'}")

    # Save training info (without optimizer state)
    info = {
        "epoch": mx.array(epoch),
        "loss": mx.array(loss),
        "learning_rate": mx.array(optimizer.learning_rate),
    }
    info_path = os.path.join(save_dir, f"info_epoch_{epoch}.npz")
    mx.savez(info_path, **info)

    return model_path, info_path


def load_checkpoint(model, optimizer, checkpoint_dir, epoch):
    """Load model checkpoint"""
    model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.safetensors")
    info_path = os.path.join(checkpoint_dir, f"info_epoch_{epoch}.npz")

    # Load model weights
    model.load_weights(model_path)

    # Load training info
    info = mx.load(info_path)
    epoch = int(info["epoch"].item())
    loss = float(info["loss"].item())
    optimizer.learning_rate = float(info["learning_rate"].item())

    return epoch, loss


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Training")
    parser.add_argument(
        "--mode",
        type=str,
        default="dev",
        choices=["dev", "full"],
        help="Training mode: dev (local development) or full (full training)",
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
        default=None,
        help="Override default number of epochs",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="./VOCdevkit/VOC2012",
        help="Path to VOC dataset",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Training settings based on mode
    if args.mode == "dev":
        # Development settings
        batch_size = args.batch_size or 4
        num_epochs = args.epochs or 100
        val_frequency = 20
        train_size = 100
        val_size = 20
    else:
        # Full training settings
        batch_size = args.batch_size or 16
        num_epochs = args.epochs or 125
        val_frequency = 50
        train_size = None  # Use full dataset
        val_size = None

    learning_rate = 1e-4

    # Create datasets
    train_dataset = VOCDataset(
        data_dir=args.data_dir,
        year="2012",
        image_set="train",
        augment=args.mode == "full",  # Enable augmentation for full training
    )

    if train_size:
        train_dataset.image_ids = train_dataset.image_ids[:train_size]

    val_dataset = VOCDataset(
        data_dir=args.data_dir,
        year="2012",
        image_set="val",
        augment=False,
    )

    if val_size:
        val_dataset.image_ids = val_dataset.image_ids[:val_size]

    print(f"\nTraining Configuration:")
    print(f"Mode: {args.mode}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Training Images: {len(train_dataset)}")
    print(f"Validation Images: {len(val_dataset)}")
    print(f"Data Augmentation: {args.mode == 'full'}\n")

    train_loader = create_data_loader(train_dataset, batch_size=batch_size)
    val_loader = create_data_loader(val_dataset, batch_size=batch_size)

    # Training loop
    print("Initializing model...")
    model = YOLO()
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Table setup
    headers = ["Epoch", "Loss", "XY Loss", "WH Loss", "Val Loss", "Time(s)", "Best"]
    table = []

    # Print fixed header
    print("\033[H\033[J")  # Clear screen
    print("\033[s")  # Save cursor position
    print(tabulate([], headers=headers, tablefmt="grid"))
    header_lines = len(headers) + 2  # Account for grid lines
    print(f"\033[{header_lines}A")  # Move cursor up to prepare for data

    best_val_loss = float("inf")
    last_val_loss = "N/A"  # Store last validation loss

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_xy_loss = 0
        epoch_wh_loss = 0
        num_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Training step
            loss, components = train_step(model, batch, optimizer)

            # Evaluate immediately to free memory
            mx.eval(loss, components)

            epoch_loss += loss.item()
            epoch_xy_loss += components["xy"]
            epoch_wh_loss += components["wh"]
            num_batches += 1

        # Calculate epoch metrics
        avg_loss = epoch_loss / num_batches
        avg_xy_loss = epoch_xy_loss / num_batches
        avg_wh_loss = epoch_wh_loss / num_batches
        epoch_time = time.time() - start_time

        # Only run validation at specified frequency
        if (epoch + 1) % val_frequency == 0:
            val_loss = validate(model, val_loader)
            last_val_loss = f"{val_loss:.4f}"
            is_best = val_loss < best_val_loss

            if is_best:
                best_val_loss = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    val_loss,
                    os.path.join("checkpoints", "best"),
                )
        else:
            is_best = False

        # Save latest model at end of training
        if epoch + 1 == num_epochs:
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                avg_loss,
                os.path.join("checkpoints", "latest"),
            )

        # Add row to table
        row = [
            f"{epoch + 1}/{num_epochs}",
            f"{avg_loss:.4f}",
            f"{avg_xy_loss:.4f}",
            f"{avg_wh_loss:.4f}",
            last_val_loss,
            f"{epoch_time:.2f}",
            "*" if is_best else "",
        ]
        table.append(row)

        # Update table display
        print("\033[u")  # Restore cursor to header position
        print("\033[J")  # Clear screen below cursor
        print(tabulate(table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
