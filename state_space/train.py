import sys
import os
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from data.speech_commands_loader import create_speech_commands_loaders
from model import S4Model


def run_sampling_evaluation(model, train_loader, num_samples=10, verbose=True):
    """Run sampling evaluation on training data

    Returns:
        Dictionary with per-class accuracy and overall accuracy
    """
    class_correct = {}
    class_total = {}

    for y in range(num_samples):
        for x, labels in train_loader.create_batches(
            batch_size=1, shuffle=True
        ):
            logits = model(x)
            final_logits = logits[:, -1, :]  # Shape: (batch, num_classes)
            predicted = mx.argmax(final_logits, axis=1)

            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                true_class = train_loader.classes[true_label]
                pred_class = train_loader.classes[pred_label]
                confidence = mx.softmax(final_logits[i])[pred_label].item()
                status = "✓" if true_label == pred_label else "✗"

                # Track class accuracy
                if true_class not in class_total:
                    class_total[true_class] = 0
                    class_correct[true_class] = 0
                class_total[true_class] += 1
                if true_label == pred_label:
                    class_correct[true_class] += 1

                if verbose:
                    print(
                        f"Sample {y+1}: {status} True: {true_class} | Pred: {pred_class} (conf: {confidence:.3f})"
                    )
            break

    # Calculate overall accuracy
    total_correct = sum(class_correct.values())
    total_samples = sum(class_total.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

    results = {
        'class_correct': class_correct,
        'class_total': class_total,
        'overall_accuracy': overall_accuracy
    }

    if verbose:
        print("\nPer-class accuracy:")
        for cls in sorted(class_total.keys()):
            acc = class_correct.get(cls, 0) / class_total[cls]
            print(f"  {cls}: {acc:.1%} ({class_correct.get(cls, 0)}/{class_total[cls]})")
        print(f"\nOverall accuracy: {overall_accuracy:.1%} ({total_correct}/{total_samples})")

    return results


def train_speech_recognition(overfit_mode=False, init_strategy="standard"):
    """Train state space model on Google Speech Commands dataset

    Args:
        overfit_mode: Whether to run in overfit mode with 3 samples
        init_strategy: Initialization strategy - 'standard', 'improved', or 'hippo'
    """

    # Hyperparameters
    if overfit_mode:
        batch_size = 3  # Small batch for overfitting
        num_epochs = 600  # More epochs to ensure overfitting
        print(f"Running in overfit mode - using only 3 samples with {init_strategy} initialization")
    else:
        batch_size = 32
        num_epochs = 10
        print(f"Using {init_strategy} initialization strategy")

    learning_rate = 1e-3
    dim_state = 64
    n_layers = 3

    # Data parameters
    use_spectrogram = True
    n_mels = 80

    print("Loading Speech Commands dataset...")
    # Create data loaders
    train_loader, val_loader, test_loader = create_speech_commands_loaders(
        data_dir="data/speech_commands_v2",
        use_spectrogram=use_spectrogram,
        n_mels=n_mels,
        background_noise_prob=0.1,
        background_noise_volume=0.1,
        overfit_mode=overfit_mode,
        overfit_samples=3,
    )

    # Get dimensions from a sample batch
    sample_features, _ = next(train_loader.create_batches(batch_size=1))
    seq_len, dim_input = sample_features.shape[1], sample_features.shape[2]
    num_classes = len(train_loader.classes)

    print(f"Input dimensions: seq_len={seq_len}, dim_input={dim_input}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_loader.classes}")

    # Create model
    model = S4Model(
        dim_input=dim_input,
        dim_state=dim_state,
        dim_output=num_classes,
        n_layers=n_layers,
        init_strategy=init_strategy,
    )

    optimizer = optim.Adam(learning_rate=learning_rate)

    @mx.compile
    def loss_fn(params, x, y):
        """Cross-entropy loss for classification"""
        # Set model parameters
        model.update(params)

        # Forward pass through state space model
        logits = model(x)  # Shape: (batch, seq_len, num_classes)

        # Take last timestep for classification (most informative)
        final_logits = logits[:, -1, :]  # Shape: (batch, num_classes)

        # Cross-entropy loss
        loss = nn.losses.cross_entropy(final_logits, y)

        return mx.mean(loss)

    def evaluate(model, data_loader):
        """Evaluate model accuracy"""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        for features, labels in data_loader.create_batches(
            batch_size=batch_size, shuffle=False
        ):
            # Forward pass
            logits = model(features)
            pooled_logits = logits[:, -1, :]

            # Loss
            loss = nn.losses.cross_entropy(pooled_logits, labels)
            total_loss += mx.mean(loss).item()

            # Accuracy
            predicted = mx.argmax(pooled_logits, axis=1)
            correct = mx.sum(predicted == labels).item()
            total_correct += correct
            total_samples += len(labels)
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return avg_loss, accuracy

    # Training loop
    print("Starting training...")

    best_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 50  # Stop if no improvement for 50 epochs
    early_stop_threshold = 0.01  # Stop if loss below this
    high_loss_threshold = 1.0  # Cancel if loss still above this after 300 epochs

    progress_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    for epoch in progress_bar:
        total_loss = 0.0
        num_batches = 0

        # Learning rate scheduling for overfit mode
        if overfit_mode and epoch > 0:
            # Reduce learning rate if loss hasn't improved much in the last few epochs
            if epoch % 50 == 0:
                new_lr = optimizer.learning_rate * 0.9
                optimizer.learning_rate = new_lr
            #    print(f"  Reduced learning rate to {new_lr:.6f}")

        # Normal training with shuffling
        for features, labels in train_loader.create_batches(
            batch_size=batch_size, shuffle=not overfit_mode
        ):
            # Forward and backward pass
            loss, grads = mx.value_and_grad(loss_fn)(
                model.parameters(), features, labels
            )

            # Calculate gradient norm for monitoring
            grad_norm = 0.0
            for g in grads.values():
                if isinstance(g, dict):
                    for gg in g.values():
                        if isinstance(gg, mx.array):
                            grad_norm += mx.sum(gg * gg).item()
                elif isinstance(g, mx.array):
                    grad_norm += mx.sum(g * g).item()
            grad_norm = grad_norm ** 0.5

            # Update parameters
            optimizer.update(model, grads)
            mx.eval(model.parameters())

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar with current batch loss and gradient norm
            current_loss = loss.item() if "loss" in locals() else 0.0
            if overfit_mode and epoch % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{current_loss:.4f}", "GradNorm": f"{grad_norm:.3f}"})
            else:
                progress_bar.set_postfix({"Loss": f"{current_loss:.4f}"})

        # Calculate average loss for epoch
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        # Early stopping logic for overfit mode
        if overfit_mode:
            # Check for early stopping success (low loss)
            if avg_loss < early_stop_threshold:
                print(f"\n✓ Early stopping: Loss {avg_loss:.4f} < {early_stop_threshold} at epoch {epoch+1}")
                print("\nRunning final evaluation:")
                run_sampling_evaluation(model, train_loader, num_samples=10, verbose=True)
                break

            # Check for failure to converge after 300 epochs
            if epoch >= 300 and avg_loss > high_loss_threshold:
                print(f"\n✗ Stopping: Loss {avg_loss:.4f} still > {high_loss_threshold} after {epoch+1} epochs")
                print("Model failed to converge. Try a different initialization or hyperparameters.")
                break

            # Track best loss for patience-based early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Stop if no improvement for too long (but only after giving it a fair chance)
            if epoch > 100 and patience_counter >= early_stop_patience:
                print(f"\n→ Early stopping: No improvement for {early_stop_patience} epochs (best loss: {best_loss:.4f})")
                break

        # Show detailed logging for overfit mode at the last epoch
        if overfit_mode and epoch == num_epochs - 1:
            print("\nFinal evaluation:")
            run_sampling_evaluation(model, train_loader, num_samples=10, verbose=True)

    # Final test evaluation
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("Final Test Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")

    return model, train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train state space model on speech commands"
    )
    parser.add_argument(
        "--overfit", action="store_true", help="Run in overfit mode with 3 samples"
    )
    parser.add_argument(
        "--init",
        type=str,
        default="standard",
        choices=["standard", "improved", "hippo"],
        help="Initialization strategy: standard, improved, or hippo (default: standard)"
    )

    args = parser.parse_args()

    model, train_loader, val_loader, test_loader = train_speech_recognition(
        overfit_mode=args.overfit,
        init_strategy=args.init
    )
