import os
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from model import YOLO
from loss import yolo_loss
from data.voc import VOCDataset, create_data_loader, VOC_CLASSES
import json
import math


def save_checkpoint(model: YOLO, optimizer, epoch, loss, save_dir):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    try:
        save_path = os.path.join(save_dir, f"yolo_epoch_{epoch}.npz")
        model.save_weights(save_path)
        print(f"Successfully saved model weights to {save_path}")
    except Exception as e:
        print(f"Error saving model weights: {str(e)}")
        raise

    # Save model config separately
    try:
        config_path = os.path.join(save_dir, f"yolo_config_{epoch}.json")
        config = {
            "S": model.S,
            "B": model.B,
            "C": model.C,
            "anchors": model.anchors.tolist(),
        }
        with open(config_path, "w") as f:
            json.dump(config, f)
        print(f"Successfully saved model config to {config_path}")
    except Exception as e:
        print(f"Error saving model config: {str(e)}")
        raise

    # Save optimizer state
    try:
        save_path = os.path.join(save_dir, f"optimizer_epoch_{epoch}.npz")
        save_dict = {
            "learning_rate": mx.array(optimizer.learning_rate, dtype=mx.float32),
            "step": mx.array(optimizer.state.get("step", 0), dtype=mx.int32),
        }
        mx.savez(save_path, **save_dict)
        print(f"Successfully saved optimizer state to {save_path}")
    except Exception as e:
        print(f"Error saving optimizer state: {str(e)}")
        raise

    # Save training info
    info = {"epoch": epoch, "loss": loss, "model_config": config}
    info_path = os.path.join(save_dir, f"info_epoch_{epoch}.json")
    with open(info_path, "w") as f:
        json.dump(info, f)
    print(f"Successfully saved training info to {info_path}")


def load_checkpoint(model: YOLO, optimizer, checkpoint_dir, epoch):
    """Load model checkpoint"""
    try:
        # Load and verify model config
        config_path = os.path.join(checkpoint_dir, f"yolo_config_{epoch}.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        if config["S"] != model.S or config["B"] != model.B or config["C"] != model.C:
            raise ValueError(
                f"Model config mismatch. Checkpoint has S={config['S']}, "
                f"B={config['B']}, C={config['C']}, but model has S={model.S}, "
                f"B={model.B}, C={model.C}"
            )

        # Load model weights
        model_path = os.path.join(checkpoint_dir, f"yolo_epoch_{epoch}.npz")
        model.load_weights(model_path)
        print(f"Successfully loaded model weights from {model_path}")

        # Load optimizer state
        optimizer_path = os.path.join(checkpoint_dir, f"optimizer_epoch_{epoch}.npz")
        print(f"\nLoading optimizer state from {optimizer_path}")
        opt_dict = mx.load(optimizer_path)

        # Update optimizer with saved state
        optimizer.learning_rate = float(opt_dict["learning_rate"])
        if "step" in opt_dict:
            optimizer.state["step"] = int(opt_dict["step"])

        print(f"Loaded learning rate: {optimizer.learning_rate}")
        if "step" in opt_dict:
            print(f"Loaded step count: {optimizer.state['step']}")

        # Load training info
        info_path = os.path.join(checkpoint_dir, f"info_epoch_{epoch}.json")
        with open(info_path, "r") as f:
            info = json.load(f)

        print(f"\nSuccessfully loaded checkpoint from epoch {epoch}")
        print(f"Previous loss: {info['loss']:.4f}")

        return epoch, info["loss"]
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        raise


def clip_gradients(gradients, max_norm: float = 10.0):
    """Clip gradients by global norm"""
    # Compute total norm across all gradient values in the dictionary
    total_norm_sq = 0.0
    for grad in gradients.values():
        if isinstance(grad, mx.array):
            total_norm_sq += mx.sum(grad * grad).item()
    total_norm = mx.sqrt(total_norm_sq)

    # Compute scaling factor as a scalar
    clip_coef = float(max_norm / (total_norm + 1e-6))
    clip_coef = min(clip_coef, 1.0)

    # Scale all gradients with scalar coefficient
    return {
        k: g * mx.array(clip_coef) if isinstance(g, mx.array) else g
        for k, g in gradients.items()
    }


def compute_class_weights(dataset):
    """Compute class weights based on class frequency"""
    class_counts = mx.zeros(len(VOC_CLASSES))
    total_objects = 0
    
    for _, target in dataset:
        # target shape: [S, S, B*(5+C)]
        target = mx.reshape(target, (dataset.grid_size, dataset.grid_size, -1, 5 + len(VOC_CLASSES)))
        obj_mask = target[..., 4:5]  # Object confidence
        class_labels = target[..., 5:]  # Class labels
        
        # Count objects per class
        for i in range(len(VOC_CLASSES)):
            class_counts[i] += mx.sum(obj_mask * class_labels[..., i:i+1])
        total_objects += mx.sum(obj_mask)
    
    # Compute weights (inverse frequency)
    class_weights = total_objects / (class_counts + 1e-6)
    # Normalize weights
    class_weights = class_weights / mx.sum(class_weights) * len(VOC_CLASSES)
    return class_weights


def compute_box_iou(pred_boxes, target_boxes):
    """Compute IoU between predicted and target bounding boxes"""
    # Calculate intersection area
    x1 = mx.maximum(pred_boxes[..., 0], target_boxes[..., 0])
    y1 = mx.maximum(pred_boxes[..., 1], target_boxes[..., 1])
    x2 = mx.minimum(pred_boxes[..., 2], target_boxes[..., 2])
    y2 = mx.minimum(pred_boxes[..., 3], target_boxes[..., 3])
    intersection = mx.maximum(x2 - x1, 0) * mx.maximum(y2 - y1, 0)

    # Calculate union area
    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    target_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
    union = pred_area + target_area - intersection

    # Compute IoU
    iou = intersection / (union + 1e-6)
    return iou


def cosine_warmup_schedule(epoch, warmup_epochs, total_epochs, initial_lr):
    """Learning rate schedule with warmup and cosine decay"""
    if epoch < warmup_epochs:
        # Linear warmup
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return initial_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train_step(model, batch, optimizer):
    """Single training step with proper gradient computation"""
    images, targets = batch

    # Define loss function
    def loss_fn():
        predictions = model(images)
        loss, components = yolo_loss(predictions, targets, model)
        return loss, components

    # Compute loss and gradients
    (loss, components), grads = nn.value_and_grad(loss_fn, has_aux=True)()

    # Update model parameters
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    return loss, components


def train(
    data_dir: str,
    save_dir: str,
    num_epochs: int = 135,
    batch_size: int = 32,
    accumulation_steps: int = 2,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    resume_epoch: int | None = None,
    max_grad_norm: float = 5.0,
    grid_size: int = 7,
    warmup_epochs: int = 5,
    val_freq: int = 1,
    lambda_coord: float = 5.0,
    lambda_noobj: float = 1.0,
):
    """Train YOLO model with improved training process"""
    # Create model and optimizer
    model = YOLO(S=grid_size, B=2, C=len(VOC_CLASSES))

    optimizer = optim.Adam(
        learning_rate=learning_rate,
        betas=[beta1, beta2],
        eps=epsilon
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if resume_epoch is not None:
        print(f"Resuming from epoch {resume_epoch}")
        start_epoch, prev_loss = load_checkpoint(
            model, optimizer, save_dir, resume_epoch
        )
        print(f"Resumed from epoch {start_epoch} with loss {prev_loss:.4f}")

    # Verify model works with dummy input
    dummy_input = mx.random.normal((1, 448, 448, 3))
    _ = model(dummy_input)
    print("Model successfully verified with dummy input")

    # Load datasets
    print("Loading datasets...")
    train_dataset = VOCDataset(
        data_dir=data_dir,
        year="2012",
        image_set="train",
        augment=True,
        grid_size=grid_size,
    )
    val_dataset = VOCDataset(
        data_dir=data_dir,
        year="2012",
        image_set="val",
        augment=False,
        grid_size=grid_size,
    )
    
    train_loader = create_data_loader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = create_data_loader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # Compute class weights from training data
    print("Computing class weights...")
    class_weights = compute_class_weights(train_dataset)
    
    # Apply additional weighting to common classes
    class_weights = mx.array(class_weights)
    for class_name, weight in zip(VOC_CLASSES, class_weights):
        if class_name in ['person', 'chair', 'car']:
            class_weights[VOC_CLASSES.index(class_name)] *= 1.5  # Increase weight for common classes
    
    # Normalize weights again
    class_weights = class_weights / mx.sum(class_weights) * len(VOC_CLASSES)
    print("Adjusted class weights:", [f"{VOC_CLASSES[i]}: {w:.2f}" for i, w in enumerate(class_weights)])

    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_metrics = {
            'loss': 0,
            'coord': 0,
            'conf': 0,
            'noobj': 0,
            'class': 0,
            'iou': 0  # Track IoU as additional metric
        }
        num_batches = 0
        start_time = time.time()

        # Adjust learning rate
        current_lr = cosine_warmup_schedule(
            epoch, warmup_epochs, num_epochs, learning_rate
        )
        optimizer.learning_rate = current_lr
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Learning Rate: {current_lr:.6f}")

        # Training
        for batch in train_loader:
            # Training step with gradient accumulation
            accumulated_loss = 0
            accumulated_components = None
            accumulated_iou = 0

            for _ in range(accumulation_steps):
                # Forward pass and compute gradients
                loss, components = train_step(model, batch, optimizer)
                
                # Calculate IoU for monitoring
                predictions = model(batch[0])
                pred_boxes = predictions[..., :4]
                target_boxes = batch[1][..., :4]
                iou = compute_box_iou(pred_boxes, target_boxes)
                mean_iou = mx.mean(iou).item()
                
                accumulated_loss += loss.item() / accumulation_steps
                accumulated_iou += mean_iou / accumulation_steps
                
                if accumulated_components is None:
                    accumulated_components = {
                        k: v / accumulation_steps for k, v in components.items()
                    }
                else:
                    for k, v in components.items():
                        accumulated_components[k] += v / accumulation_steps

            # Update metrics
            epoch_metrics['loss'] += accumulated_loss
            epoch_metrics['iou'] += accumulated_iou
            for k, v in accumulated_components.items():
                epoch_metrics[k] += v
            num_batches += 1

            # Print batch progress with IoU
            if num_batches % 10 == 0:
                print(
                    f"Batch [{num_batches}], "
                    f"Loss: {accumulated_loss:.4f}, "
                    f"IoU: {accumulated_iou:.4f}, "
                    f"Coord: {accumulated_components['coord']:.4f}, "
                    f"Conf: {accumulated_components['conf']:.4f}, "
                    f"Class: {accumulated_components['class']:.4f}, "
                    f"NoObj: {accumulated_components['noobj']:.4f}"
                )

        # Calculate epoch metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches

        # Validation
        if (epoch + 1) % val_freq == 0:
            model.eval()
            val_metrics = {
                'loss': 0,
                'coord': 0,
                'conf': 0,
                'noobj': 0,
                'class': 0
            }
            num_val_batches = 0

            print("\nRunning validation...")
            for val_batch in val_loader:
                # Forward pass only
                predictions = model(val_batch[0])
                val_loss, val_components = yolo_loss(
                    predictions, val_batch[1], model,
                    class_weights=class_weights
                )

                # Update validation metrics
                val_metrics['loss'] += val_loss.item()
                for k, v in val_components.items():
                    val_metrics[k] += v
                num_val_batches += 1

            # Calculate validation metrics
            for k in val_metrics:
                val_metrics[k] /= num_val_batches

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_checkpoint(model, optimizer, epoch + 1, val_metrics['loss'], save_dir)
                print(f"New best model saved! Validation loss: {best_val_loss:.4f}")

        # Print epoch summary
        time_taken = time.time() - start_time
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"Time: {time_taken:.2f}s")
        print("\nTraining Metrics:")
        for k, v in epoch_metrics.items():
            print(f"{k.capitalize()}: {v:.4f}")
        
        if (epoch + 1) % val_freq == 0:
            print("\nValidation Metrics:")
            for k, v in val_metrics.items():
                print(f"{k.capitalize()}: {v:.4f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch + 1, epoch_metrics['loss'], save_dir)
            print(f"Checkpoint saved at epoch {epoch+1}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument(
        "--data-dir", default="./VOCdevkit/VOC2012", help="Path to VOC dataset"
    )
    parser.add_argument(
        "--save-dir", default="./checkpoints", help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=135, help="Number of epochs to train"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=2,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="Beta1 for Adam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="Beta2 for Adam optimizer"
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-8, help="Epsilon for Adam optimizer"
    )
    parser.add_argument(
        "--resume-epoch", type=int, help="Resume training from this epoch"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=5.0, help="Maximum gradient norm"
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=7,
        help="Grid size for YOLO (S x S grid)",
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=5, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--val-freq", type=int, default=1, help="Validation frequency"
    )
    parser.add_argument(
        "--lambda-coord", type=float, default=5.0, help="Lambda for coordinate loss"
    )
    parser.add_argument(
        "--lambda-noobj", type=float, default=1.0, help="Lambda for no object loss"
    )
    args = parser.parse_args()

    config = {
        "data_dir": args.data_dir,
        "save_dir": args.save_dir,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "learning_rate": args.learning_rate,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "epsilon": args.epsilon,
        "resume_epoch": args.resume_epoch,
        "max_grad_norm": args.max_grad_norm,
        "grid_size": args.grid_size,
        "warmup_epochs": args.warmup_epochs,
        "val_freq": args.val_freq,
        "lambda_coord": args.lambda_coord,
        "lambda_noobj": args.lambda_noobj,
    }

    train(**config)
