import os
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from model import YOLO
from loss import yolo_loss
from data.voc import VOCDataset, create_data_loader
import json
from mlx.utils import tree_flatten, tree_map


def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    try:
        save_path = os.path.join(save_dir, f"yolo_epoch_{epoch}.npz")

        # Save flattened parameters
        flat_params = tree_flatten(model.parameters())

        print("\nModel parameters after saving: ")
        for k, v in flat_params:
            print(f"{k}: {v.shape}, mean: {mx.mean(v):.4f}")

        mx.savez(save_path, **dict(flat_params))
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


def load_checkpoint(model, optimizer, checkpoint_dir, epoch):
    """Load model checkpoint"""
    try:
        # Print initial parameter stats
        print("\nModel parameters before loading:")
        current_params = tree_flatten(model.parameters())
        for k, v in current_params:
            print(f"{k}: {v.shape}, mean: {mx.mean(v):.4f}")

        # Load model weights
        model_path = os.path.join(checkpoint_dir, f"yolo_epoch_{epoch}.npz")
        print(f"\nLoading model from {model_path}")
        model_state = mx.load(model_path)

        # Print loaded parameters
        print("\nLoaded parameters from checkpoint:")
        for k, v in model_state.items():
            print(f"{k}: {v.shape}, mean: {mx.mean(v):.4f}")

        # Create parameter dictionary preserving the order from tree_flatten
        ordered_params = {}
        for k, _ in current_params:
            if k not in model_state:
                raise ValueError(f"Parameter {k} not found in checkpoint")
            ordered_params[k] = model_state[k]

        # Update model with ordered parameters and evaluate
        model.update(ordered_params)
        mx.eval(model.parameters())  # Force evaluation of the update

        # Verify parameters after update
        print("\nModel parameters after update:")
        updated_params = tree_flatten(model.parameters())
        for k, v in updated_params:
            v_eval = mx.eval(v)  # Evaluate each parameter
            if v_eval is not None:
                print(f"{k}: {v_eval.shape}, mean: {mx.mean(v_eval):.4f}")

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
    last_loss = None
    if resume_epoch is not None:
        start_epoch, last_loss = load_checkpoint(
            model, optimizer, save_dir, resume_epoch
        )
        print(f"Resumed from epoch {start_epoch} with loss {last_loss:.4f}")

        # Ensure model is in training mode after loading
        model.train(True)

        # Verify model state with dummy input in NHWC format
        dummy_input = mx.zeros((1, 448, 448, 3))  # (batch, height, width, channels)
        try:
            _ = model(dummy_input)
            mx.eval(_)
            print("Model successfully verified with dummy input")
        except Exception as e:
            print(f"Error verifying model: {str(e)}")
            raise

    # Create dataset and data loader
    print("Loading dataset...")
    dataset = VOCDataset(data_dir, year="2012", image_set="train")
    val_dataset = VOCDataset(data_dir, year="2012", image_set="val")

    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, num_epochs):
        model.train(True)  # Set to training mode
        epoch_loss = 0.0
        start_time = time.time()

        # Get training batches
        train_images, train_targets = create_data_loader(
            dataset, batch_size, shuffle=True
        )

        accumulated_grads = None
        # Train for one epoch
        for batch_idx, (images, targets) in enumerate(zip(train_images, train_targets)):

            def loss_fn(params):
                model.update(params)
                predictions = model(images)
                return yolo_loss(predictions, targets)

            # Compute loss and gradients
            loss, grads = mx.value_and_grad(loss_fn)(model.parameters())

            # Scale gradients for accumulation
            grads = tree_map(lambda x: x / accumulation_steps, grads)

            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(
                    lambda x, y: x + y, accumulated_grads, grads
                )

            # Update weights after accumulation steps
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.update(model, accumulated_grads)
                accumulated_grads = None
                mx.eval(model.parameters())  # Force evaluation to free memory

            epoch_loss += loss.item()

            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Batch [{batch_idx+1}/{len(train_images)}], "
                    f"Loss: {loss.item():.4f}"
                )

            # Force memory cleanup
            if (batch_idx + 1) % 5 == 0:
                mx.eval(model.parameters())

        # Update with any remaining accumulated gradients
        if accumulated_grads is not None:
            optimizer.update(model, accumulated_grads)
            mx.eval(model.parameters())

        # Compute epoch statistics
        avg_loss = epoch_loss / len(train_images)
        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Average Loss: {avg_loss:.4f}, "
            f"Time: {epoch_time:.2f}s"
        )

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs - 1:
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
    }

    train(**config)
