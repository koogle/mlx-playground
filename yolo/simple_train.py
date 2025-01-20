import os
import mlx.core as mx
import mlx.optimizers as optim
from tqdm import tqdm
from model import YOLO
from data.voc import VOC_CLASSES, DataLoader, VOCDataset, create_data_loader
import time
from tabulate import tabulate
import argparse
from pathlib import Path
from loss import yolo_loss
import psutil  # Add this import
import gc  # Add this import
import logging  # Add this import
from datetime import datetime
import traceback  # Add this import


def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return log_file


def get_memory_usage():
    """Get current memory usage information"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        "rss": mem_info.rss / (1024 * 1024),  # RSS in MB
        "vms": mem_info.vms / (1024 * 1024),  # VMS in MB
        "system_percent": psutil.virtual_memory().percent,
    }


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
    """Single training step optimized for MLX"""
    images, targets = batch

    try:
        mem_before = get_memory_usage()
        logging.debug(f"Memory before training step: {mem_before}")

        @mx.compile
        def loss_fn(params, images, targets):
            model.update(params)
            predictions = model(images)
            return yolo_loss(predictions, targets, model)

        # Compute loss and gradients
        loss, grads = mx.value_and_grad(loss_fn)(model.parameters(), images, targets)

        optimizer.update(model, grads)

        # Evaluate model state and loss together for better batching
        state = [model.parameters(), optimizer.state]
        loss_val, components = loss
        mx.eval(loss_val, *state)

        mem_after = get_memory_usage()
        logging.debug(f"Memory after training step: {mem_after}")

        # Memory change logging
        mem_diff = {k: mem_after[k] - mem_before[k] for k in mem_before}
        logging.debug(f"Memory change during step: {mem_diff}")

        # Clear intermediate values
        del grads, state
        gc.collect()  # Force garbage collection

        return loss_val, components

    except Exception as e:
        logging.error(f"\nError in training step:")
        logging.error(f"Exception type: {type(e).__name__}")
        logging.error(f"Exception message: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        logging.error(f"Memory state: {get_memory_usage()}")
        logging.error(
            f"Batch shapes - Images: {images.shape}, Targets: {targets.shape}"
        )
        raise


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
        default=48,
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


def train_epoch(model, train_loader, optimizer):
    """Train for one epoch with MLX optimizations"""
    model.train()
    epoch_losses = {"total": 0, "xy": 0, "wh": 0, "conf": 0, "class": 0, "iou": 0}
    num_batches = 0
    start_time = time.time()

    logging.info(f"\nStarting epoch with {len(train_loader)} batches")
    initial_memory = get_memory_usage()
    logging.info(f"Initial memory state: {initial_memory}")

    for batch_idx, batch in enumerate(train_loader):
        try:
            # Log memory before batch
            if batch_idx % 10 == 0:  # Log every 10 batches
                logging.info(
                    f"Memory state before batch {batch_idx}: {get_memory_usage()}"
                )

            # Training step
            loss, components = train_step(model, batch, optimizer)

            # Update metrics after successful step
            current_loss = loss.item()
            epoch_losses["total"] += current_loss
            for k in ["xy", "wh", "conf", "class", "iou"]:
                if k in components:
                    epoch_losses[k] += components[k]
            num_batches += 1

            # Print progress less frequently
            if batch_idx % 20 == 0:
                avg_loss = epoch_losses["total"] / num_batches
                elapsed_time = time.time() - start_time
                mem_usage = get_memory_usage()
                logging.info(
                    f"\nBatch {batch_idx}/{len(train_loader)} "
                    f"(Loss: {current_loss:.4f}, Avg: {avg_loss:.4f}, "
                    f"Time: {elapsed_time:.1f}s)\n"
                    f"Memory Usage: {mem_usage}"
                )

            # Explicit cleanup
            del loss, components
            if batch_idx % 10 == 0:  # Periodic garbage collection
                gc.collect()

        except Exception as e:
            logging.error(f"\nError in batch {batch_idx}:")
            logging.error(f"Exception: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            logging.error(f"Memory state: {get_memory_usage()}")
            continue

    # Calculate averages
    for k in epoch_losses:
        epoch_losses[k] /= max(num_batches, 1)

    final_memory = get_memory_usage()
    logging.info(f"Final memory state: {final_memory}")

    return epoch_losses, time.time() - start_time


def main():
    args = parse_args()
    log_file = setup_logging()
    logging.info(f"Starting training with log file: {log_file}")
    logging.info(f"Command line arguments: {args}")

    print("\nInitializing training...")
    print(f"Mode: {args.mode}")
    print(f"Data directory: {args.data_dir}")

    # Create datasets with size limits for dev mode
    print("\nCreating datasets...")
    train_dataset = VOCDataset(args.data_dir, "train")
    val_dataset = VOCDataset(args.data_dir, "val")

    # Limit dataset size in dev mode
    if args.mode == "dev":
        # Take only first 10 images for dev mode
        dev_size = 10
        train_dataset.image_ids = train_dataset.image_ids[:dev_size]
        val_dataset.image_ids = val_dataset.image_ids[:dev_size]
        print(f"Dev mode: Limited to {dev_size} images")

    # Modify batch size selection
    if args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = 2 if args.mode == "dev" else 48

    logging.info(f"Using batch size: {batch_size}")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Expected batches per epoch: {len(train_dataset) // batch_size}")

    # Create data loaders
    train_loader = create_data_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = create_data_loader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Verify data loaders
    print("\nVerifying data loaders...")
    try:
        first_batch = next(iter(train_loader))
        images, targets = first_batch
        print(
            f"First train batch shapes - Images: {images.shape}, Targets: {targets.shape}"
        )

        first_val_batch = next(iter(val_loader))
        val_images, val_targets = first_val_batch
        print(
            f"First val batch shapes - Images: {val_images.shape}, Targets: {val_targets.shape}"
        )
    except Exception as e:
        print(f"Error loading batches: {str(e)}")
        raise

    # Initialize model and optimizer
    model = YOLO(S=7, B=5, C=20)
    optimizer = optim.Adam(learning_rate=1e-4)

    # Training loop
    val_frequency = 5  # Validate every 5 epochs
    best_val_loss = float("inf")
    last_val_loss = float("inf")

    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Training
        epoch_losses, epoch_time = train_epoch(model, train_loader, optimizer)

        # Print batch-level progress
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {epoch_losses['total']:.4f}")
        print(f"Components: {format_loss_components(epoch_losses)}")
        print(f"Time: {epoch_time:.1f}s")

        # Validation
        if (epoch + 1) % val_frequency == 0:
            print("\nRunning validation...")
            val_loss = validate(model, val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch + 1, "best")
                print("New best model saved!")
            last_val_loss = val_loss

        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch + 1, "regular")

    try:
        # ... existing training loop ...
        pass
    except Exception as e:
        logging.error("Fatal error in training:")
        logging.error(traceback.format_exc())
        logging.error(f"Final memory state: {get_memory_usage()}")
        raise


if __name__ == "__main__":
    main()
