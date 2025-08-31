import os
import mlx.core as mx


def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"diffusion_epoch_{epoch}.npz")
    model.save_weights(model_path)
    print(f"Saved model to {model_path}")

    optim_path = os.path.join(save_dir, f"optimizer_epoch_{epoch}.npz")
    mx.savez(
        optim_path,
        learning_rate=mx.array(optimizer.learning_rate),
        step=mx.array(optimizer.state.get("step", 0)),
    )
    print(f"Saved optimizer state to {optim_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    # Load model weights
    model_path = f"{checkpoint_path}.npz"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    # Extract epoch from filename if possible
    try:
        epoch = int(os.path.basename(checkpoint_path).split("_")[-1])
    except:
        raise ValueError(f"Invalid checkpoint filename: {checkpoint_path}")

    model.load_weights(model_path)
    print(f"Loaded model from {model_path}")

    # Try to load optimizer state
    optim_path = checkpoint_path.replace("diffusion_", "optimizer_")
    optim_path = f"{optim_path}.npz"
    if os.path.exists(optim_path):
        optim_state = mx.load(optim_path)
        if "learning_rate" in optim_state:
            optimizer.learning_rate = float(optim_state["learning_rate"])
        print(f"Loaded optimizer state from {optim_path}")
    else:
        print("Optimizer state not found")

    return epoch
