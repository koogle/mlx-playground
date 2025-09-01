import os
import json
import signal
import time
import sys
import mlx.core as mx


def get_checkpoint_name(epoch, model_type="unconditional"):
    """
    Generate consistent checkpoint naming across save and load operations
    
    Args:
        epoch: Epoch number
        model_type: "unconditional" or "conditional"
    
    Returns:
        Base checkpoint name (without extension)
    """
    prefix = "conditional" if model_type == "conditional" else "diffusion"
    return f"{prefix}_checkpoint_epoch_{epoch}"


def parse_checkpoint_name(checkpoint_name):
    """
    Parse checkpoint name to extract epoch and model type
    
    Args:
        checkpoint_name: Checkpoint filename (with or without extension)
    
    Returns:
        Tuple of (epoch, model_type) or (None, None) if parsing fails
    """
    # Remove extension if present
    if checkpoint_name.endswith(".npz"):
        checkpoint_name = checkpoint_name[:-4]
    if checkpoint_name.endswith("_metadata.json"):
        checkpoint_name = checkpoint_name[:-14]
    if checkpoint_name.endswith("_optimizer.npz"):
        checkpoint_name = checkpoint_name[:-14]
    
    # Try to parse the checkpoint name
    if "_checkpoint_epoch_" in checkpoint_name:
        try:
            parts = checkpoint_name.split("_checkpoint_epoch_")
            prefix = parts[0]
            epoch = int(parts[1].split("_")[0])  # Handle any suffix
            
            # Determine model type from prefix
            if prefix == "conditional":
                model_type = "conditional"
            elif prefix == "diffusion":
                model_type = "unconditional"
            else:
                model_type = "unknown"
            
            return epoch, model_type
        except:
            pass
    
    # Try old format
    if checkpoint_name.startswith("checkpoint_epoch_"):
        try:
            epoch = int(checkpoint_name.split("_")[2])
            return epoch, "unconditional"  # Assume old format is unconditional
        except:
            pass
    
    return None, None


def save_checkpoint(
    model, optimizer, epoch, loss, save_dir, model_type="unconditional", config=None
):
    """
    Save model checkpoint with metadata

    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        loss: Current loss value
        save_dir: Directory to save checkpoint
        model_type: "unconditional" or "conditional"
        config: Optional configuration dictionary to save
    """
    os.makedirs(save_dir, exist_ok=True)

    # Use consistent checkpoint naming
    checkpoint_name = get_checkpoint_name(epoch, model_type)

    # Save model weights
    model_path = os.path.join(save_dir, f"{checkpoint_name}.npz")
    model.save_weights(model_path)
    print(f"Saved {model_type} model to {model_path}")

    # Save optimizer state
    optim_path = os.path.join(save_dir, f"{checkpoint_name}_optimizer.npz")
    mx.savez(
        optim_path,
        learning_rate=mx.array(optimizer.learning_rate),
        step=mx.array(optimizer.state.get("step", 0)),
    )

    # Save metadata
    metadata = {
        "epoch": epoch,
        "loss": float(loss),
        "model_type": model_type,
        "model_path": model_path,
        "optimizer_path": optim_path,
    }

    # Add model-specific metadata
    if model_type == "conditional":
        metadata["num_classes"] = getattr(model, "num_classes", 10)
        metadata["class_emb_dim"] = getattr(model, "class_emb_dim", 128)

    # Add config if provided
    if config is not None:
        metadata["config"] = config

    metadata_path = os.path.join(save_dir, f"{checkpoint_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved checkpoint metadata to {metadata_path}")
    return checkpoint_name


def load_checkpoint(model, optimizer, checkpoint_path, expected_type=None):
    """
    Load model checkpoint with validation

    Args:
        model: The model to load weights into
        optimizer: The optimizer to load state into
        checkpoint_path: Path to checkpoint (can be directory or specific file)
        expected_type: Optional expected model type ("unconditional" or "conditional")

    Returns:
        epoch: The epoch number from the checkpoint
    """
    # Handle different path formats
    if os.path.isdir(checkpoint_path):
        # Find the latest checkpoint in directory
        checkpoint_path = find_latest_checkpoint_in_dir(checkpoint_path)
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_path}")

    # Remove .npz extension if present
    if checkpoint_path.endswith(".npz"):
        checkpoint_path = checkpoint_path[:-4]

    # Check for metadata file first
    metadata_path = f"{checkpoint_path}_metadata.json"
    metadata = {}

    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Validate model type if specified
        if expected_type and metadata.get("model_type") != expected_type:
            print(
                f"Warning: Loading {metadata.get('model_type', 'unknown')} "
                f"checkpoint but expected {expected_type}"
            )

        model_path = metadata.get("model_path", f"{checkpoint_path}.npz")
        optim_path = metadata.get("optimizer_path", f"{checkpoint_path}_optimizer.npz")
        epoch = metadata.get("epoch", 0)

        print(
            f"Loading {metadata.get('model_type', 'unknown')} model from epoch {epoch}"
        )
        print(f"Last loss: {metadata.get('loss', 'unknown')}")
    else:
        # Fallback to old format
        model_path = f"{checkpoint_path}.npz"

        # Try to extract epoch and model type from filename
        parsed_epoch, parsed_type = parse_checkpoint_name(os.path.basename(checkpoint_path))
        if parsed_epoch is not None:
            epoch = parsed_epoch
            if expected_type and parsed_type != expected_type and parsed_type != "unknown":
                print(f"Warning: Loading {parsed_type} checkpoint but expected {expected_type}")
        else:
            epoch = 0
            print(f"Warning: Could not extract epoch from {checkpoint_path}")

        # Guess optimizer path
        optim_path = f"{checkpoint_path}_optimizer.npz"
        if not os.path.exists(optim_path):
            # Try old format
            optim_path = checkpoint_path.replace("checkpoint_", "optimizer_")
            optim_path = f"{optim_path}.npz"

    # Load model weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_weights(model_path)
    print(f"Loaded model from {model_path}")

    # Load optimizer state
    if os.path.exists(optim_path):
        optim_state = mx.load(optim_path)
        if "learning_rate" in optim_state:
            optimizer.learning_rate = float(optim_state["learning_rate"])
        print(f"Loaded optimizer state from {optim_path}")
    else:
        print("Warning: Optimizer state not found")

    return epoch


def find_latest_checkpoint_in_dir(checkpoint_dir):
    """
    Find the latest checkpoint in a directory

    Args:
        checkpoint_dir: Directory to search for checkpoints

    Returns:
        Path to latest checkpoint (without extension) or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = []

    # Look for checkpoint files with metadata
    for file in os.listdir(checkpoint_dir):
        if file.endswith("_metadata.json"):
            try:
                with open(os.path.join(checkpoint_dir, file), "r") as f:
                    metadata = json.load(f)
                epoch = metadata.get("epoch", 0)
                base_name = file.replace("_metadata.json", "")
                checkpoints.append((epoch, os.path.join(checkpoint_dir, base_name)))
            except:
                continue

    # Fallback: Look for .npz files and parse their names
    if not checkpoints:
        for file in os.listdir(checkpoint_dir):
            if file.endswith(".npz") and "optimizer" not in file:
                epoch, model_type = parse_checkpoint_name(file)
                if epoch is not None:
                    base_name = file.replace(".npz", "")
                    checkpoints.append((epoch, os.path.join(checkpoint_dir, base_name)))

    if checkpoints:
        # Return checkpoint with highest epoch
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints[-1][1]

    return None


def list_checkpoints(checkpoint_dir):
    """
    List all available checkpoints with their metadata

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        List of checkpoint info dictionaries
    """
    if not os.path.exists(checkpoint_dir):
        return []

    checkpoints = []

    for file in os.listdir(checkpoint_dir):
        if file.endswith("_metadata.json"):
            try:
                with open(os.path.join(checkpoint_dir, file), "r") as f:
                    metadata = json.load(f)
                metadata["filename"] = file.replace("_metadata.json", "")
                checkpoints.append(metadata)
            except:
                continue

    # Sort by epoch
    checkpoints.sort(key=lambda x: x.get("epoch", 0))

    return checkpoints


class InterruptHandler:
    """
    Graceful interrupt handler with double Ctrl+C force quit

    Usage:
        handler = InterruptHandler(save_callback)
        handler.setup()

        # In training loop:
        if handler.interrupted:
            break
    """

    def __init__(self, save_callback=None, patience=3.0):
        """
        Args:
            save_callback: Function to call on first interrupt (e.g., save checkpoint)
            patience: Seconds to wait for second Ctrl+C before force quit
        """
        self.save_callback = save_callback
        self.patience = patience
        self.interrupted = False
        self.first_interrupt_time = None
        self.save_completed = False
        self.original_handler = None

    def handle_interrupt(self, signum, frame):
        """Handle interrupt signal"""
        current_time = time.time()

        # First interrupt - try to save gracefully
        if not self.interrupted:
            self.interrupted = True
            self.first_interrupt_time = current_time

            print("\n\n========================================")
            print("🛑 Interrupt received! Saving progress...")
            print("Press Ctrl+C again to force quit (without saving)")
            print("========================================\n")

            # Call save callback if provided
            if self.save_callback and not self.save_completed:
                try:
                    self.save_callback()
                    self.save_completed = True
                    print("\n✅ Progress saved successfully!")
                    print("Exiting gracefully...")
                except Exception as e:
                    print(f"\n⚠️ Error during save: {e}")
                    print("Press Ctrl+C again to force quit")

        # Second interrupt within patience window - force quit
        elif current_time - self.first_interrupt_time < self.patience:
            print("\n\n========================================")
            print("⚠️  FORCE QUIT - Exiting immediately!")
            print("Warning: Progress may not be saved")
            print("========================================\n")

            # Restore original handler and exit
            if self.original_handler is not None:
                signal.signal(signal.SIGINT, self.original_handler)

            sys.exit(1)

        # Second interrupt after patience window - treat as new interrupt
        else:
            self.first_interrupt_time = current_time
            print(
                "\n🛑 Another interrupt received. Press Ctrl+C again quickly to force quit."
            )

    def setup(self):
        """Set up the signal handler"""
        self.original_handler = signal.signal(signal.SIGINT, self.handle_interrupt)
        return self

    def cleanup(self):
        """Restore original signal handler"""
        if self.original_handler is not None:
            signal.signal(signal.SIGINT, self.original_handler)

    def __enter__(self):
        """Context manager entry"""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
        return False

    @property
    def should_stop(self):
        """Check if training should stop"""
        return self.interrupted and self.save_completed


def create_interrupt_handler(
    model,
    optimizer,
    epoch_getter,
    loss_getter,
    checkpoint_dir,
    model_type="unconditional",
    config=None,
):
    """
    Create an interrupt handler with checkpoint saving

    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch_getter: Function that returns current epoch
        loss_getter: Function that returns current loss
        checkpoint_dir: Directory to save checkpoint
        model_type: Type of model ("unconditional" or "conditional")
        config: Configuration to save

    Returns:
        InterruptHandler instance
    """

    def save_callback():
        """Save checkpoint on interrupt"""
        epoch = epoch_getter()
        loss = loss_getter()

        save_checkpoint(
            model,
            optimizer,
            epoch,
            loss,
            checkpoint_dir,
            model_type=model_type,
            config=config,
        )

        print(f"Checkpoint saved at epoch {epoch}")
        print("Run the script again to resume from this checkpoint.")

    return InterruptHandler(save_callback)
