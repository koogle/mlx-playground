import os
import mlx.core as mx
import mlx.optimizers as optim
from model import YOLO
from data.voc import VOC_CLASSES, DataLoader, VOCDataset, create_data_loader
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
    """Single training step with memory optimizations"""
    images, targets = batch

    def loss_fn(params, images, targets):
        model.update(params)
        predictions = model(images)
        return yolo_loss(predictions, targets, model)

    loss, grads = mx.value_and_grad(loss_fn)(model.parameters(), images, targets)
    optimizer.update(model, grads)

    # Evaluate loss and components
    loss_val, components = loss
    mx.eval(loss_val)

    # Clear intermediate values
    del grads
    return loss_val, components


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

    for batch in val_loader:
        images, targets = batch
        predictions = model(images)
        loss, components = yolo_loss(predictions, targets, model)

        # Accumulate losses properly
        val_losses["total"] += loss.item()
        for k in components:
            if k in val_losses:  # Only accumulate basic components
                val_losses[k] += components[k]
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
        default=125,
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


def train_epoch(model, train_loader, optimizer, epoch, show_batches=False):
    """Train for one epoch"""
    model.train()
    epoch_losses = {"total": 0, "xy": 0, "wh": 0, "conf": 0, "class": 0, "iou": 0}
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        # Training step
        loss, components = train_step(model, batch, optimizer)

        # Update metrics
        epoch_losses["total"] += loss.item()
        for k in ["xy", "wh", "conf", "class", "iou"]:
            if k in components:
                epoch_losses[k] += components[k]
        num_batches += 1

        # Ensure evaluation of tensors
        mx.eval(loss)
        del loss, components  # Explicitly delete to help with memory

        if show_batches and batch_idx % 10 == 0:
            print(
                f"Batch {batch_idx}: loss={epoch_losses['total']/max(1,num_batches):.4f}"
            )

    # Calculate averages
    for k in epoch_losses:
        epoch_losses[k] /= num_batches

    return epoch_losses, time.time() - start_time


def main():
    args = parse_args()

    # Create datasets
    train_dataset = VOCDataset(args.data_dir, "train")
    val_dataset = VOCDataset(args.data_dir, "val")

    # Create data loaders
    train_loader = create_data_loader(
        dataset=train_dataset, batch_size=args.batch_size or 32, shuffle=True
    )

    val_loader = create_data_loader(
        dataset=val_dataset, batch_size=args.batch_size or 32, shuffle=False
    )

    # Initialize model and optimizer
    model = YOLO(S=7, B=5, C=20)
    optimizer = optim.Adam(learning_rate=1e-4)

    # Training loop
    val_frequency = 5  # Validate every 5 epochs
    best_val_loss = float("inf")
    last_val_loss = float("inf")

    for epoch in range(args.epochs):
        # Training
        epoch_losses, epoch_time = train_epoch(model, train_loader, optimizer, epoch)

        # Validation
        if (epoch + 1) % val_frequency == 0:
            val_loss = validate(model, val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch + 1, "best")
                print("New best model saved!")
            last_val_loss = val_loss
        else:
            val_loss = last_val_loss

        # Print progress
        print(f"\nEpoch {epoch + 1}:")
        print(f"Train Loss: {epoch_losses['total']:.4f}")
        print(f"Components: {format_loss_components(epoch_losses)}")
        print(f"Time: {epoch_time:.1f}s")


if __name__ == "__main__":
    main()
