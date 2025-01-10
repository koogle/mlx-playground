import os
import mlx.core as mx
import mlx.optimizers as optim
from model import YOLO
from data.voc import VOC_CLASSES, VOCDataset, create_data_loader
import time
from tabulate import tabulate
import argparse
from pathlib import Path
from loss import yolo_loss


def bbox_loss(predictions, targets, model):
    """YOLO loss focusing on box regression using IoU"""
    batch_size = predictions.shape[0]
    S = model.S  # Grid size (e.g., 7x7)
    B = model.B  # Number of boxes per cell

    # Reshape predictions and targets
    pred_boxes = mx.reshape(
        predictions[..., : B * 4], (batch_size, S, S, B, 4), allowzero=True
    )
    pred_conf = mx.sigmoid(predictions[..., B * 4 : (B * 5)])
    pred_conf = mx.reshape(pred_conf, (batch_size, S, S, B))

    # Get target boxes and object mask
    targ = mx.reshape(targets[..., :4], (batch_size, S, S, 1, 4))
    targ = mx.repeat(targ, B, axis=3)  # Repeat target for each predictor
    obj_mask = mx.reshape(targets[..., 4:5], (batch_size, S, S, 1))
    obj_mask = mx.repeat(obj_mask, B, axis=3)

    # Create anchor points grid
    grid_x, grid_y = mx.meshgrid(
        mx.arange(S, dtype=mx.float32), mx.arange(S, dtype=mx.float32)
    )
    grid_xy = mx.stack([grid_x, grid_y], axis=-1)  # [S,S,2]
    grid_xy = mx.expand_dims(grid_xy, axis=2)  # [S,S,1,2]
    grid_xy = mx.broadcast_to(grid_xy, (batch_size, S, S, B, 2))  # [batch,S,S,B,2]

    # Get predicted box coordinates
    pred_xy = grid_xy / S  # Anchor points in [0,1] space
    pred_wh = pred_boxes[..., 2:4]  # Direct width/height prediction

    # Combine into final boxes
    pred_boxes_global = mx.concatenate([pred_xy, pred_wh], axis=-1)

    # Calculate IoU between predictions and targets
    ious = calculate_iou(pred_boxes_global, targ)

    # Find best predictor for each object
    best_ious = mx.max(ious, axis=-1, keepdims=True)
    best_box_mask = (ious == best_ious) * obj_mask

    # Compute losses only for cells with objects
    num_objects = mx.sum(obj_mask) + 1e-6

    # Box regression loss using IoU
    loc_loss = mx.sum(best_box_mask * (1 - ious)) / num_objects

    # Confidence loss
    conf_loss_obj = mx.sum(best_box_mask * mx.square(pred_conf - ious)) / num_objects
    conf_loss_noobj = mx.sum((1 - obj_mask) * mx.square(pred_conf)) / (
        batch_size * S * S * B - num_objects + 1e-6
    )

    # Total loss with weighted components
    total_loss = loc_loss + conf_loss_obj + 0.1 * conf_loss_noobj

    if True:  # Debug prints
        print(f"Loc Loss: {loc_loss.item():.4f}")
        print(f"Conf Obj Loss: {conf_loss_obj.item():.4f}")
        print(f"Conf NoObj Loss: {conf_loss_noobj.item():.4f}")
        print(f"Num Objects: {num_objects.item()}")

        # Print some prediction details
        obj_indices = mx.where(obj_mask > 0)
        if len(obj_indices[0]) > 0:
            b, i, j, k = [x[0] for x in obj_indices]
            print("\nExample prediction:")
            print(f"Target box: {targ[b,i,j,k].tolist()}")
            print(f"Pred box: {pred_boxes_global[b,i,j,k].tolist()}")
            print(f"IoU: {ious[b,i,j,k].item():.4f}")

    return total_loss, (loc_loss, conf_loss_obj, num_objects)


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
    """Single training step"""
    images, targets = batch

    def loss_fn(params, images, targets):
        model.update(params)
        predictions = model(images)  # [batch, S, S, B*(5+C)]
        return yolo_loss(predictions, targets, model)

    loss, grads = mx.value_and_grad(loss_fn)(model.parameters(), images, targets)
    optimizer.update(model, grads)
    return loss


def format_loss_components(components):
    """Format loss components for display"""
    return (
        f"XY: {components['xy']:.4f}, "
        f"WH: {components['wh']:.4f}, "
        f"Conf: {components['conf']:.4f}, "
        f"Class: {components['class']:.4f}, "
        f"IoU: {components['iou']:.4f}"
    )


def validate(model, val_loader):
    """Run validation with aggregated loss components"""
    model.eval()
    val_losses = {"total": 0, "xy": 0, "wh": 0, "conf": 0, "class": 0, "iou": 0}
    num_batches = 0
    print("\nValidation:")

    for batch in val_loader:
        images, targets = batch
        predictions = model(images)

        # Debug prediction statistics
        analyze_predictions(predictions, targets, model)

        loss, components = yolo_loss(predictions, targets, model)

        # Accumulate losses
        val_losses["total"] += loss.item()
        for k, v in components.items():
            val_losses[k] += v
        num_batches += 1

    # Calculate averages
    for k in val_losses:
        val_losses[k] /= num_batches

    # Print validation summary
    print(f"\nValidation Summary:")
    print(f"Total Loss: {val_losses['total']:.4f}")
    print(f"Components: {format_loss_components(val_losses)}")

    return val_losses["total"]


def save_checkpoint(model, optimizer, epoch, checkpoint_type="regular"):
    """Save model checkpoint

    Args:
        model: YOLO model instance
        optimizer: Optimizer instance
        epoch: Current epoch number
        checkpoint_type: Either "regular" or "best"
    """
    # Create checkpoint directory
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    # Determine checkpoint path based on type
    if checkpoint_type == "best":
        model_path = save_dir / "best_model.safetensors"
        info_path = save_dir / "best_info.npz"
    else:
        model_path = save_dir / f"model_epoch_{epoch}.safetensors"
        info_path = save_dir / f"info_epoch_{epoch}.npz"

    print(f"\nSaving {checkpoint_type} checkpoint to {model_path}")

    # Save model weights
    model.save_weights(str(model_path))

    # Save training info
    info = {
        "epoch": mx.array(epoch),
        "learning_rate": mx.array(optimizer.learning_rate),
    }
    mx.savez(str(info_path), **info)

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


def analyze_predictions(predictions, targets, model):
    """Analyze prediction statistics for debugging"""
    batch_size = predictions.shape[0]
    S = model.S
    B = model.B
    C = model.C

    # Reshape predictions
    pred = mx.reshape(predictions, (batch_size, S, S, B, 5 + C))

    # Extract components
    pred_xy = mx.sigmoid(pred[..., 0:2])  # [batch,S,S,B,2]
    pred_wh = pred[..., 2:4]  # [batch,S,S,B,2]
    pred_conf = mx.sigmoid(pred[..., 4])  # [batch,S,S,B]

    # Extract targets
    target_xy = targets[..., 0:2]  # [batch,S,S,2]
    target_wh = targets[..., 2:4]  # [batch,S,S,2]
    target_conf = targets[..., 4]  # [batch,S,S]

    print("\nPrediction Analysis:")
    print(f"XY range: [{pred_xy.min().item():.4f}, {pred_xy.max().item():.4f}]")
    print(f"WH range: [{pred_wh.min().item():.4f}, {pred_wh.max().item():.4f}]")
    print(f"Conf range: [{pred_conf.min().item():.4f}, {pred_conf.max().item():.4f}]")

    print("\nTarget Analysis:")
    print(f"XY range: [{target_xy.min().item():.4f}, {target_xy.max().item():.4f}]")
    print(f"WH range: [{target_wh.min().item():.4f}, {target_wh.max().item():.4f}]")
    print(
        f"Conf range: [{target_conf.min().item():.4f}, {target_conf.max().item():.4f}]"
    )

    # Count objects
    num_objects = mx.sum(target_conf).item()
    print(f"\nNumber of objects: {num_objects}")


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
        train_dataset.image_ids = dev_image_ids
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
    model = YOLO(S=7, B=5, C=20)
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
        epoch_losses = {"total": 0, "xy": 0, "wh": 0, "conf": 0, "class": 0, "iou": 0}
        num_batches = 0
        start_time = time.time()

        if show_batches:
            print(f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, batch in enumerate(train_loader):
            # Training step
            loss, components = train_step(model, batch, optimizer)

            # Print batch details if enabled
            # if show_batches:
            #    print(f"\nBatch {batch_idx}:")
            #    print(f"Total Loss: {loss.item():.4f}")
            #    print(f"Components: {format_loss_components(components)}")

            # Update epoch metrics
            epoch_losses["total"] += loss.item()
            for k, v in components.items():
                epoch_losses[k] += v
            num_batches += 1

            # Evaluate immediately to free memory
            mx.eval(loss)

        # Calculate epoch metrics
        epoch_time = time.time() - start_time
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        # Validation
        if (epoch + 1) % val_frequency == 0:
            val_loss = validate(model, val_loader)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch + 1, "best")
                print("New best model saved!")
        else:
            val_loss = last_val_loss

        # Add row to table
        table.append(
            [
                epoch + 1,
                f"{epoch_losses['total']:.4f}",
                f"{epoch_losses['xy']:.4f}",
                f"{epoch_losses['wh']:.4f}",
                f"{epoch_losses['conf']:.4f}",
                f"{epoch_losses['class']:.4f}",
                f"{epoch_losses['iou']:.4f}",
                f"{val_loss:.4f}" if isinstance(val_loss, float) else val_loss,
                f"{epoch_time:.1f}",
                "*" if val_loss == best_val_loss else "",
            ]
        )

        # Print progress
        if not show_batches or (epoch + 1) % 10 == 0:
            print("\nTraining Progress:")
            print(
                tabulate(
                    table[-10:],  # Show last 10 epochs
                    headers=[
                        "Epoch",
                        "Loss",
                        "XY",
                        "WH",
                        "Conf",
                        "Class",
                        "IoU",
                        "Val",
                        "Time",
                        "Best",
                    ],
                    tablefmt="simple",
                )
            )

        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch + 1)

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
