import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from model import YOLO
from data.voc import VOCDataset, create_data_loader
import time


def bbox_loss(predictions, targets, model):
    """
    Simplified loss focusing only on bounding box regression.
    Ignores class predictions and only considers coordinate and confidence loss.
    """
    batch_size = predictions.shape[0]
    S = model.S  # Grid size
    B = model.B  # Number of boxes per cell

    # Reshape predictions and targets to [batch, S, S, B, 5]
    # Where 5 is (x, y, w, h, confidence)
    pred = mx.reshape(predictions[..., : B * 5], (batch_size, S, S, B, 5))
    targ = mx.reshape(targets[..., :5], (batch_size, S, S, 1, 5))
    targ = mx.repeat(targ, B, axis=3)  # Repeat for each predicted box

    # Extract components
    pred_xy = pred[..., :2]  # Center coordinates
    pred_wh = pred[..., 2:4]  # Width and height
    pred_conf = mx.sigmoid(pred[..., 4:5])  # Confidence score

    targ_xy = targ[..., :2]
    targ_wh = targ[..., 2:4]
    obj_mask = targ[..., 4:5]  # Object presence mask
    noobj_mask = 1 - obj_mask

    # Coordinate loss (only for cells with objects)
    coord_loss = obj_mask * (
        mx.sum(mx.square(pred_xy - targ_xy), axis=-1)  # xy loss
        + mx.sum(
            mx.square(mx.sqrt(pred_wh + 1e-6) - mx.sqrt(targ_wh + 1e-6)), axis=-1
        )  # wh loss
    )
    coord_loss = 5.0 * mx.sum(coord_loss)  # Higher weight for coordinate loss

    # Confidence loss
    conf_loss = obj_mask * mx.square(
        pred_conf - 1
    ) + 0.5 * noobj_mask * mx.square(  # Object confidence loss
        pred_conf
    )  # No object confidence loss
    conf_loss = mx.sum(conf_loss)

    # Total loss
    total_loss = (coord_loss + conf_loss) / batch_size

    # Return loss components for monitoring
    components = {
        "coord": coord_loss.item() / batch_size,
        "conf": conf_loss.item() / batch_size,
    }

    return total_loss, components


def train_step(model, batch, optimizer):
    """Single training step with simplified bbox-only loss"""
    images, targets = batch

    def loss_fn(params):
        model.update(params)
        predictions = model(images)
        loss, components = bbox_loss(predictions, targets, model)
        return loss, components

    # Compute loss and gradients
    (loss, components), grads = mx.value_and_grad(loss_fn, has_aux=True)(
        model.parameters()
    )

    # Update parameters
    optimizer.update(model, grads)

    # Ensure updates are processed
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


def main():
    # Training settings
    batch_size = 4  # Even smaller batch size for debugging
    num_epochs = 20  # More epochs since task is simpler
    learning_rate = 5e-5  # Slightly lower learning rate
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
    # Use only 10 images for validation
    val_dataset.image_ids = val_dataset.image_ids[:10]

    train_loader = create_data_loader(train_dataset, batch_size=batch_size)
    val_loader = create_data_loader(val_dataset, batch_size=batch_size)

    # Create model and optimizer
    print("Initializing model...")
    model = YOLO()  # Using default settings
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Training loop
    print("\nStarting training...")
    print("Focus: Bounding box regression only (no classification)")
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        start_time = time.time()

        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        for batch_idx, batch in enumerate(train_loader):
            # Training step
            loss, components = train_step(model, batch, optimizer)
            epoch_loss += loss.item()
            num_batches += 1

            # Print progress every batch
            print(
                f"Batch {batch_idx + 1}/{len(train_loader)}, "
                f"Loss: {loss.item():.4f}, "
                f"Coord Loss: {components['coord']:.4f}, "
                f"Conf Loss: {components['conf']:.4f}"
            )

        # Calculate epoch metrics
        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Time: {epoch_time:.2f}s")

        # Validation
        if (epoch + 1) % val_frequency == 0:
            val_loss = validate(model, val_loader)
            print(f"Validation Loss: {val_loss:.4f}")

            # Save if best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("New best model! Saving checkpoint...")
                os.makedirs("checkpoints", exist_ok=True)
                mx.savez("checkpoints/best_bbox_model.npz", **model.parameters())


if __name__ == "__main__":
    main()
