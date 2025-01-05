import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from model import YOLO
from data.voc import VOCDataset, create_data_loader
import time


def bbox_loss(predictions, targets, model):
    """
    Simplified loss focusing only on bounding box coordinate regression.
    Only computes coordinate loss (x, y, w, h) without confidence scores.
    """
    batch_size = predictions.shape[0]
    S = model.S  # Grid size
    B = model.B  # Number of boxes per cell

    # Reshape predictions and targets to [batch, S, S, B, 4] (only x,y,w,h)
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

    # Coordinate losses
    xy_loss = mx.sum(mx.square(pred_xy - targ_xy), axis=-1)  # Position loss
    wh_loss = mx.sum(
        mx.square(mx.sqrt(mx.maximum(pred_wh, 0) + 1e-6) - mx.sqrt(targ_wh + 1e-6)),
        axis=-1,
    )  # Size loss

    # Only compute loss for cells that contain objects
    coord_loss = obj_mask * (xy_loss + wh_loss)
    total_loss = (
        5.0 * mx.sum(coord_loss) / batch_size
    )  # Higher weight for coordinate loss

    # Return components for monitoring
    components = {
        "xy": mx.sum(obj_mask * xy_loss).item() / batch_size,
        "wh": mx.sum(obj_mask * wh_loss).item() / batch_size,
    }

    return total_loss, components


def train_step(model, batch, optimizer):
    """Single training step with coordinate-only loss"""
    images, targets = batch

    def loss_fn(params, images, targets):
        model.update(params)
        predictions = model(images)
        return bbox_loss(predictions, targets, model)

    # Compute loss and gradients
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    (loss, components), grads = loss_and_grad_fn(model.parameters(), images, targets)

    # Update parameters
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    return loss, components


def validate(model, val_loader):
    """Run validation"""
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in val_loader:
        images, targets = batch
        predictions = model(images)
        loss, _ = bbox_loss(predictions, targets, model)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights using safetensors
    model_path = os.path.join(save_dir, f"model_epoch_{epoch}.safetensors")
    model.save_weights(model_path)

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


def main():
    # Training settings
    batch_size = 4
    num_epochs = 20
    learning_rate = 1e-4  # Slightly increased for coordinate-only training
    val_frequency = 1

    # Create a very small subset of data for testing
    train_dataset = VOCDataset(
        data_dir="./VOCdevkit/VOC2012",
        year="2012",
        image_set="train",
        augment=False,
    )

    # Use only 50 images for initial testing
    train_dataset.image_ids = train_dataset.image_ids[:50]

    val_dataset = VOCDataset(
        data_dir="./VOCdevkit/VOC2012",
        year="2012",
        image_set="val",
        augment=False,
    )
    val_dataset.image_ids = val_dataset.image_ids[:10]

    train_loader = create_data_loader(train_dataset, batch_size=batch_size)
    val_loader = create_data_loader(val_dataset, batch_size=batch_size)

    # Create model and optimizer
    print("Initializing model...")
    model = YOLO()
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Training loop
    print("\nStarting training...")
    print("Focus: Bounding box coordinate regression only")
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_xy_loss = 0
        epoch_wh_loss = 0
        num_batches = 0
        start_time = time.time()

        print(f"\nEpoch {epoch + 1}/{num_epochs}", end="\r")

        for batch_idx, batch in enumerate(train_loader):
            # Training step
            loss, components = train_step(model, batch, optimizer)
            epoch_loss += loss.item()
            epoch_xy_loss += components["xy"]
            epoch_wh_loss += components["wh"]
            num_batches += 1
            # Print minimal progress
            # print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Calculate epoch metrics
        avg_loss = epoch_loss / num_batches
        avg_xy_loss = epoch_xy_loss / num_batches
        avg_wh_loss = epoch_wh_loss / num_batches
        epoch_time = time.time() - start_time

        # print(f"\nEpoch {epoch + 1} Summary:")
        print(f"\nAverage Loss: {avg_loss:.4f}")
        print(f"Average XY Loss: {avg_xy_loss:.4f}")
        print(f"Average WH Loss: {avg_wh_loss:.4f}")
        print(f"Time: {epoch_time:.2f}s")

        # Validation
        if (epoch + 1) % val_frequency == 0:
            val_loss = validate(model, val_loader)
            print(f"Validation Loss: {val_loss:.4f}")

            # Save if best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("New best model! Saving checkpoint...")
                save_checkpoint(model, optimizer, epoch + 1, val_loss, "checkpoints")


if __name__ == "__main__":
    main()
