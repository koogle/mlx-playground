import os
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from model import YOLO
from loss import yolo_loss
from data.voc import VOCDataset, create_data_loader
import json


def save_checkpoint(model: YOLO, optimizer, epoch, loss, save_dir):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    try:
        save_path = os.path.join(save_dir, f"yolo_epoch_{epoch}.npz")
        model.save_weights(save_path)
        print(f"Successfully saved model to {save_path}")
    except Exception as e:
        print(f"Error saving model state: {str(e)}")
        raise

    # Save optimizer state
    try:
        save_path = os.path.join(save_dir, f"optimizer_epoch_{epoch}.npz")
        # Only save the learning rate and step count
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
    info = {
        "epoch": epoch,
        "loss": loss,
    }
    info_path = os.path.join(save_dir, f"info_epoch_{epoch}.json")
    with open(info_path, "w") as f:
        json.dump(info, f)
    print(f"Successfully saved training info to {info_path}")


def load_checkpoint(model: YOLO, optimizer, checkpoint_dir, epoch):
    """Load model checkpoint"""
    try:
        model_path = os.path.join(checkpoint_dir, f"yolo_epoch_{epoch}.npz")
        model.load_weights(model_path)

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
    total_norm = mx.sqrt(sum(mx.sum(g * g) for g in gradients))
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = mx.minimum(clip_coef, 1.0)
    return [g * clip_coef for g in gradients]


def adjust_learning_rate(optimizer, epoch, initial_lr):
    """Adjust learning rate using step decay"""
    # Decay learning rate by 0.1 at epochs 60 and 90
    if epoch == 60:
        optimizer.learning_rate = initial_lr * 0.1
    elif epoch == 90:
        optimizer.learning_rate = initial_lr * 0.01
    return optimizer.learning_rate


def train(
    data_dir: str,
    save_dir: str,
    num_epochs: int = 135,
    batch_size: int = 8,
    accumulation_steps: int = 2,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    resume_epoch: int | None = None,
    max_grad_norm: float = 10.0,
):
    """Train YOLO model"""
    # Create model and optimizer
    print("Creating model...")
    model = YOLO()

    optimizer = optim.Adam(
        learning_rate=learning_rate, betas=[beta1, beta2], eps=epsilon
    )

    # Resume from checkpoint if specified
    start_epoch = 0
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

    # Load dataset with augmentation
    print("Loading dataset...")
    train_dataset = VOCDataset(data_dir, year="2012", image_set="train", augment=True)
    train_loader = create_data_loader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        accumulated_loss = 0.0
        accumulated_grads = None
        step_count = 0
        start_time = time.time()

        # Adjust learning rate
        current_lr = adjust_learning_rate(optimizer, epoch, learning_rate)
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Learning Rate: {current_lr:.6f}")

        for batch_idx, (images, targets) in enumerate(train_loader):
            # Forward pass
            predictions = model(images)
            loss = yolo_loss(predictions, targets)
            loss = loss / accumulation_steps  # Normalize loss for gradient accumulation
            accumulated_loss += loss

            # Compute gradients
            gradients = mx.grad(model.parameters())(predictions, targets)

            # Clip gradients
            gradients = clip_gradients(gradients, max_grad_norm)

            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = gradients
            else:
                accumulated_grads = [
                    g1 + g2 for g1, g2 in zip(accumulated_grads, gradients)
                ]

            # Update weights with accumulated gradients
            step_count += 1
            if step_count == accumulation_steps:
                optimizer.update(model, accumulated_grads)
                mx.eval(model.parameters())

                epoch_loss += accumulated_loss.item() * accumulation_steps
                batch_count += 1
                step_count = 0
                accumulated_loss = 0.0
                accumulated_grads = None

                if batch_count % 10 == 0:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], "
                        f"Batch [{batch_count}/{len(train_loader)//accumulation_steps}], "
                        f"Loss: {loss.item():.4f}"
                    )

        # Compute average loss for the epoch
        avg_loss = epoch_loss / batch_count
        end_time = time.time()
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Average Loss: {avg_loss:.4f}, "
            f"Time: {end_time - start_time:.2f}s"
        )

        # Save checkpoint every 5 epochs
        if (epoch % 5 == 0) or (epoch == num_epochs - 1):
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, save_dir)
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
        "--max-grad-norm", type=float, default=10.0, help="Maximum gradient norm"
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
    }

    train(**config)
