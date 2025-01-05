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
    """Simplified loss focusing only on bounding box coordinate regression."""
    batch_size = predictions.shape[0]
    S = model.S
    B = model.B

    # Reshape predictions and targets
    pred = mx.reshape(predictions[..., : B * 4], (batch_size, S, S, B, 4))
    targ = mx.reshape(targets[..., :4], (batch_size, S, S, 1, 4))
    targ = mx.repeat(targ, B, axis=3)

    # Extract coordinates
    pred_xy = mx.sigmoid(pred[..., :2])  # Sigmoid for relative cell position
    pred_wh = mx.sigmoid(pred[..., 2:4])  # Sigmoid for relative image size
    targ_xy = targ[..., :2]
    targ_wh = targ[..., 2:4]

    # Get object mask
    obj_mask = mx.squeeze(mx.reshape(targets[..., 4:5], (batch_size, S, S, 1)), axis=-1)
    obj_mask = mx.repeat(mx.expand_dims(obj_mask, axis=-1), B, axis=-1)

    # Position loss (MSE)
    xy_loss = mx.sum(mx.square(pred_xy - targ_xy), axis=-1)

    # Size loss (MSE)
    wh_loss = mx.sum(mx.square(pred_wh - targ_wh), axis=-1)

    # Only compute loss for cells that contain objects
    coord_loss = obj_mask * (xy_loss + wh_loss)

    # Normalize by number of objects
    num_objects = mx.sum(obj_mask) + 1e-6
    total_loss = mx.sum(coord_loss) / num_objects

    return total_loss, (xy_loss, wh_loss, num_objects)


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

    loss, (xy_loss, wh_loss, num_objects) = compiled_step()

    # Calculate mean loss components
    components = {
        "xy": mx.mean(xy_loss).item(),
        "wh": mx.mean(wh_loss).item(),
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
    # Check in the "best" folder first
    model_path = os.path.join(
        checkpoint_dir, "best", f"model_epoch_{epoch}.safetensors"
    )
    info_path = os.path.join(checkpoint_dir, "best", f"info_epoch_{epoch}.npz")

    if not os.path.exists(model_path) or not os.path.exists(info_path):
        # Fallback to "latest" folder if "best" is not available
        model_path = os.path.join(
            checkpoint_dir, "latest", f"model_epoch_{epoch}.safetensors"
        )
        info_path = os.path.join(checkpoint_dir, "latest", f"info_epoch_{epoch}.npz")

    if not os.path.exists(model_path) or not os.path.exists(info_path):
        raise FileNotFoundError(f"Checkpoint not found for epoch {epoch}")

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
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Epoch to start training from (useful for resuming)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Fixed set of image IDs for development
    dev_image_ids = [
        "2008_000008",  # Simple image with a single object
        "2008_000015",  # Multiple objects, good for testing
        "2008_000023",  # Different object sizes
        "2008_000028",  # Interesting composition
        "2008_000033",  # Multiple object classes
        "2008_000036",  # Clear object boundaries
        "2008_000042",  # Good lighting conditions
        "2008_000045",  # Multiple objects with overlap
        "2008_000052",  # Different scales
        "2008_000064",  # Good variety of objects
    ]

    # Training settings based on mode
    if args.mode == "dev":
        # Development settings - use fixed image set
        batch_size = args.batch_size or 2
        num_epochs = args.epochs or 200
        val_frequency = 10
        train_size = len(dev_image_ids)  # Use all dev images
        val_size = 2  # Use 2 images for validation
    else:
        # Full training settings
        batch_size = args.batch_size or 16
        num_epochs = args.epochs or 125
        val_frequency = 50
        train_size = None
        val_size = None

    learning_rate = 1e-4

    # Create datasets
    train_dataset = VOCDataset(
        data_dir=args.data_dir,
        year="2012",
        image_set="train",
        augment=args.mode == "full",
    )

    if args.mode == "dev":
        # Override image_ids with our fixed dev set
        train_dataset.image_ids = dev_image_ids
        print("\nDevelopment mode using fixed image set:")
        for img_id in dev_image_ids:
            print(f"  {img_id}")
    elif train_size:
        train_dataset.image_ids = train_dataset.image_ids[:train_size]

    val_dataset = VOCDataset(
        data_dir=args.data_dir,
        year="2012",
        image_set="val",
        augment=False,
    )

    if val_size:
        val_dataset.image_ids = dev_image_ids[
            :val_size
        ]  # Use first two dev images for validation

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

    # Load checkpoint if starting from a specific epoch
    if args.start_epoch > 0:
        epoch, _ = load_checkpoint(model, optimizer, "checkpoints", args.start_epoch)
        print(f"Resuming training from epoch {epoch}")
    else:
        epoch = 0

    # Adjust num_epochs to continue training for the specified number of epochs
    num_epochs += epoch

    # Table setup
    headers = ["Epoch", "Loss", "XY Loss", "WH Loss", "Val Loss", "Time(s)", "Best"]
    table = []

    best_val_loss = float("inf")
    last_val_loss = "N/A"  # Store last validation loss

    for epoch in range(epoch, num_epochs):
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

        # Print the table in a standard way
        print(tabulate(table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
