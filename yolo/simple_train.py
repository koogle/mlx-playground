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
    """YOLO loss following v2 paper approach with anchor boxes and normalized losses"""
    batch_size = predictions.shape[0]
    S = model.S  # Grid size (e.g., 7x7)
    B = model.B
    anchors = model.anchors  # Shape: (B, 2)

    # Reshape predictions and targets
    pred_boxes = mx.reshape(predictions[..., : B * 4], (batch_size, S, S, B, 4))
    pred_conf = mx.sigmoid(predictions[..., B * 4 : (B * 5)])  # Confidence scores
    pred_conf = mx.reshape(pred_conf, (batch_size, S, S, B))

    targ = mx.reshape(targets[..., :4], (batch_size, S, S, 1, 4))
    targ = mx.repeat(targ, B, axis=3)

    # Extract coordinates
    pred_xy = mx.sigmoid(pred_boxes[..., :2])  # Center coordinates [0,1] within cell

    # Reshape anchors to match prediction shape
    anchors_wh = mx.reshape(anchors, (1, 1, 1, B, 2))  # Add batch, grid, grid dims
    anchors_wh = mx.broadcast_to(anchors_wh, (batch_size, S, S, B, 2))

    # Width and height predictions
    pred_wh_raw = mx.clip(pred_boxes[..., 2:4], -4.0, 4.0)
    pred_wh = anchors_wh * mx.exp(pred_wh_raw) / S  # Scale relative to grid size

    # Target width/height are already in [0,1] range for whole image
    # Convert to grid-relative coordinates
    targ_wh = targ[..., 2:4] * S

    # Target coordinates (already in [0,1] range within cell)
    targ_xy = targ[..., :2]

    # Get object mask
    obj_mask = mx.squeeze(mx.reshape(targets[..., 4:5], (batch_size, S, S, 1)), axis=-1)
    obj_mask = mx.repeat(mx.expand_dims(obj_mask, axis=-1), B, axis=-1)
    num_objects = mx.sum(obj_mask) + 1e-6

    # Calculate IoU between predictions and targets
    pred_boxes_iou = mx.concatenate([pred_xy, pred_wh], axis=-1)
    iou_scores = calculate_iou(pred_boxes_iou, targ)

    # Find responsible predictor (highest IoU)
    best_iou_mask = (
        iou_scores == mx.max(iou_scores, axis=-1, keepdims=True)
    ) * obj_mask

    # Loss components with scaling factors
    coord_scale = 5.0
    noobj_scale = 0.5

    # Coordinate losses normalized to [0,1]
    xy_loss = (
        coord_scale * mx.sum(mx.square(pred_xy - targ_xy), axis=-1) / 2
    )  # Normalize by max possible squared difference

    wh_loss = (
        coord_scale * mx.sum(mx.square(pred_wh - targ_wh), axis=-1) / (S * S)
    )  # Normalize by grid area

    coord_loss = mx.sum(best_iou_mask * (xy_loss + wh_loss))

    # Confidence losses (already in [0,1] range due to sigmoid)
    conf_loss_obj = mx.sum(best_iou_mask * mx.square(pred_conf - iou_scores))
    conf_loss_noobj = noobj_scale * mx.sum((1 - obj_mask) * mx.square(pred_conf))

    # Total loss (normalized by number of objects)
    total_loss = (coord_loss + conf_loss_obj + conf_loss_noobj) / num_objects

    return total_loss, (xy_loss, wh_loss, mx.sum(obj_mask))


def calculate_iou(boxes1, boxes2):
    """Calculate IoU between two sets of boxes"""
    # Convert to corners format
    boxes1_x1 = boxes1[..., 0] - boxes1[..., 2] / 2
    boxes1_y1 = boxes1[..., 1] - boxes1[..., 3] / 2
    boxes1_x2 = boxes1[..., 0] + boxes1[..., 2] / 2
    boxes1_y2 = boxes1[..., 1] + boxes1[..., 3] / 2

    boxes2_x1 = boxes2[..., 0] - boxes2[..., 2] / 2
    boxes2_y1 = boxes2[..., 1] - boxes2[..., 3] / 2
    boxes2_x2 = boxes2[..., 0] + boxes2[..., 2] / 2
    boxes2_y2 = boxes2[..., 1] + boxes2[..., 3] / 2

    # Calculate intersection
    intersect_x1 = mx.maximum(boxes1_x1, boxes2_x1)
    intersect_y1 = mx.maximum(boxes1_y1, boxes2_y1)
    intersect_x2 = mx.minimum(boxes1_x2, boxes2_x2)
    intersect_y2 = mx.minimum(boxes1_y2, boxes2_y2)

    intersect_w = mx.maximum(intersect_x2 - intersect_x1, 0)
    intersect_h = mx.maximum(intersect_y2 - intersect_y1, 0)
    intersection = intersect_w * intersect_h

    # Calculate union
    boxes1_area = (boxes1_x2 - boxes1_x1) * (boxes1_y2 - boxes1_y1)
    boxes2_area = (boxes2_x2 - boxes2_x1) * (boxes2_y2 - boxes2_y1)
    union = boxes1_area + boxes2_area - intersection

    return intersection / (union + 1e-6)


def train_step(model, batch, optimizer):
    """Single training step with coordinate-only loss"""
    images, targets = batch

    def loss_fn(params, inputs, targets):
        model.update(params)
        predictions = model(inputs)
        loss, _ = bbox_loss(predictions, targets, model)
        return loss

    # Compute loss and gradients directly without compilation
    loss, grads = mx.value_and_grad(loss_fn)(model.parameters(), images, targets)

    # Monitor gradients before update
    grad_stats = {}

    def process_grad(name, g):
        if g is None:
            return
        if isinstance(g, dict):
            # Handle nested gradients (e.g. BatchNorm)
            for k, v in g.items():
                process_grad(f"{name}.{k}", v)
        else:
            try:
                g_abs = mx.abs(g)
                grad_stats[name] = {
                    "mean": mx.mean(g).item(),
                    "std": mx.std(g).item(),
                    "min": mx.min(g_abs).item(),
                    "max": mx.max(g_abs).item(),
                }
            except:
                pass

    # Process all gradients
    for name, g in grads.items():
        process_grad(name, g)

    if False:
        # Print gradient statistics
        print("\nGradient Statistics:")
        for name, stats in grad_stats.items():
            print(f"{name}:")
            print(f"  Mean: {stats['mean']:.2e}")
            print(f"  Std:  {stats['std']:.2e}")
            print(f"  Min:  {stats['min']:.2e}")
            print(f"  Max:  {stats['max']:.2e}")

    # Update model parameters
    optimizer.update(model, grads)

    # Compute loss components for logging
    _, components = bbox_loss(model(images), targets, model)
    components = {
        "xy": mx.mean(components[0]).item(),
        "wh": mx.mean(components[1]).item(),
    }

    return loss, components


def validate(model, val_loader):
    """Run validation with detailed debugging"""
    model.eval()
    losses = []
    print("\nValidation Details:")

    for batch_idx, batch in enumerate(val_loader):
        images, targets = batch
        predictions = model(images)
        loss, (xy_loss, wh_loss, num_objects) = bbox_loss(predictions, targets, model)

        # Print details for each validation image
        print(f"\nBatch {batch_idx}:")
        print(f"Number of objects: {num_objects.item()}")
        print(f"XY Loss: {mx.mean(xy_loss).item():.4f}")
        print(f"WH Loss: {mx.mean(wh_loss).item():.4f}")
        print(f"Total Loss: {loss.item():.4f}")

        # Debug predictions vs targets
        batch_size = predictions.shape[0]
        S = model.S
        B = model.B

        # Reshape and extract predictions
        pred = mx.reshape(predictions[..., : B * 4], (batch_size, S, S, B, 4))
        pred_xy = mx.sigmoid(pred[..., :2])
        pred_wh = mx.sigmoid(pred[..., 2:4])

        # Reshape and extract targets
        targ = mx.reshape(targets[..., :4], (batch_size, S, S, 1, 4))
        targ_xy = targ[..., :2]
        targ_wh = targ[..., 2:4]

        # Get object mask
        obj_mask = mx.squeeze(
            mx.reshape(targets[..., 4:5], (batch_size, S, S, 1)), axis=-1
        )

        # Find cells with objects
        for b in range(batch_size):
            print(f"\nImage {b}:")
            for i in range(S):
                for j in range(S):
                    if obj_mask[b, i, j].item() > 0:
                        print(f"Object in cell ({i},{j}):")
                        print(
                            f"  Target: xy={targ_xy[b,i,j,0].tolist()}, wh={targ_wh[b,i,j,0].tolist()}"
                        )
                        print(
                            f"  Pred 1: xy={pred_xy[b,i,j,0].tolist()}, wh={pred_wh[b,i,j,0].tolist()}"
                        )
                        print(
                            f"  Pred 2: xy={pred_xy[b,i,j,1].tolist()}, wh={pred_wh[b,i,j,1].tolist()}"
                        )

        losses.append(loss)

    # Evaluate all losses at once
    mx.eval(losses)
    avg_loss = sum(l.item() for l in losses) / len(losses)
    print(f"\nAverage validation loss: {avg_loss:.4f}")
    return avg_loss


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
    show_batches = True  # Set to True to see batch details, False for table view

    best_val_loss = float("inf")
    last_val_loss = "N/A"  # Store last validation loss

    if show_batches:
        print("Training on fixed dev set:")
        for img_id in dev_image_ids:
            print(f"  {img_id}")

    for epoch in range(epoch, num_epochs):
        model.train()
        epoch_loss = 0
        epoch_xy_loss = 0
        epoch_wh_loss = 0
        num_batches = 0
        start_time = time.time()

        if show_batches:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

        for batch_idx, batch in enumerate(train_loader):
            # Training step
            loss, components = train_step(model, batch, optimizer)

            # Print batch details if enabled
            if show_batches:
                print(f"\nBatch {batch_idx}:")
                print(f"Loss: {loss.item():.4f}")
                print(f"XY Loss: {components['xy']:.4f}")
                print(f"WH Loss: {components['wh']:.4f}")

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

        # Run validation and update table
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

        # Only print table if batch details are hidden
        if not show_batches:
            print(tabulate(table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
