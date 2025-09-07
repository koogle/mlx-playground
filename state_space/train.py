import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from data.speech_commands_loader import create_speech_commands_loaders
from model import StateSpace, S4Model


def train_speech_recognition(overfit_mode=False):
    """Train state space model on Google Speech Commands dataset"""
    
    # Hyperparameters
    if overfit_mode:
        batch_size = 3  # Small batch for overfitting
        learning_rate = 5e-4  # Lower learning rate for stable overfitting
        num_epochs = 300  # More epochs to ensure overfitting
        print("Running in overfit mode - using only 3 samples")
    else:
        batch_size = 32
        learning_rate = 1e-3
        num_epochs = 10
    
    # Model parameters  
    if overfit_mode:
        dim_state = 64  # Bigger state for more capacity
        n_layers = 3   # More layers for more capacity
    else:
        dim_state = 64
        n_layers = 3
    
    # Data parameters
    use_spectrogram = True
    n_mels = 80
    
    print("Loading Speech Commands dataset...")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_speech_commands_loaders(
        data_dir="data/speech_commands_v2",
        batch_size=batch_size,
        use_spectrogram=use_spectrogram,
        n_mels=n_mels,
        background_noise_prob=0.1,
        background_noise_volume=0.1,
        overfit_mode=overfit_mode,
        overfit_samples=3
    )
    
    # Get dimensions from a sample batch
    sample_features, sample_labels = next(train_loader.create_batches(batch_size=1))
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
        n_layers=n_layers
    )
    
    
    # Optimizer with learning rate scheduling
    if overfit_mode:
        # Use SGD with momentum for more aggressive overfitting
        optimizer = optim.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        optimizer = optim.Adam(learning_rate=learning_rate)
    
    def loss_fn(params, x, y):
        """Cross-entropy loss for classification"""
        # Set model parameters
        model.update(params)
        
        # Forward pass through state space model
        logits = model(x)  # Shape: (batch, seq_len, num_classes)
        
        # Global average pooling over time dimension for classification
        pooled_logits = mx.mean(logits, axis=1)  # Shape: (batch, num_classes)
        
        # Cross-entropy loss
        loss = nn.losses.cross_entropy(pooled_logits, y)
        # Return mean loss across batch for gradient computation
        return mx.mean(loss)
    
    def evaluate(model, data_loader):
        """Evaluate model accuracy"""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        for features, labels in data_loader.create_batches(batch_size=batch_size, shuffle=False):
            # Forward pass
            logits = model(features)
            pooled_logits = mx.mean(logits, axis=1)
            
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
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        # Learning rate scheduling for overfit mode
        if overfit_mode and epoch > 0:
            # Reduce learning rate if loss hasn't improved much in the last few epochs
            if epoch % 50 == 0:
                new_lr = optimizer.learning_rate * 0.8
                optimizer.learning_rate = new_lr
                print(f"  Reduced learning rate to {new_lr:.6f}")
        
        # Training - for overfit mode, don't shuffle and train on all 3 samples as one batch
        if overfit_mode:
            # Get all 3 samples as one batch, no shuffling
            for features, labels in train_loader.create_batches(batch_size=3, shuffle=False):
                # Forward and backward pass
                loss, grads = mx.value_and_grad(loss_fn)(model.parameters(), features, labels)
                
                # Update parameters
                optimizer.update(model, grads)
                mx.eval(model.parameters())
                
                total_loss += loss.item()
                num_batches += 1
                break  # Only train on this one batch containing all 3 samples
        else:
            # Normal training with shuffling
            for features, labels in train_loader.create_batches(batch_size=batch_size, shuffle=True):
                # Forward and backward pass
                loss, grads = mx.value_and_grad(loss_fn)(model.parameters(), features, labels)
                
                # Update parameters
                optimizer.update(model, grads)
                mx.eval(model.parameters())
                
                total_loss += loss.item()
                num_batches += 1
        
        # Show detailed logging for overfit mode
        if overfit_mode and num_batches > 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/num_batches:.4f}")
            
            # Get the same batch again for prediction display
            for features, labels in train_loader.create_batches(batch_size=3, shuffle=False):
                logits = model(features)
                pooled_logits = mx.mean(logits, axis=1)
                predicted = mx.argmax(pooled_logits, axis=1)
                
                print("  Model vs Label comparison:")
                for i in range(len(labels)):
                    true_label = labels[i].item()
                    pred_label = predicted[i].item()
                    true_class = train_loader.classes[true_label]
                    pred_class = train_loader.classes[pred_label]
                    confidence = mx.softmax(pooled_logits[i])[pred_label].item()
                    
                    status = "✓" if true_label == pred_label else "✗"
                    print(f"    Sample {i+1}: {status} True: {true_class} | Pred: {pred_class} (conf: {confidence:.3f})")
                print()
                break  # Only need the first batch
        
        # Validation
        val_loss, val_accuracy = evaluate(model, val_loader)
        avg_train_loss = total_loss / num_batches
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Check for overfitting success in overfit mode
        if overfit_mode and avg_train_loss < 0.01:
            print(f"  🎉 Successfully overfitted! Train loss: {avg_train_loss:.6f}")
            if val_accuracy == 1.0:
                print(f"  🎯 Perfect accuracy achieved on validation set!")
                break
        
        print()
    
    # Final test evaluation
    test_loss, test_accuracy = evaluate(model, test_loader)
    print(f"Final Test Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    
    return model, train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train state space model on speech commands')
    parser.add_argument('--overfit', action='store_true', help='Run in overfit mode with 3 samples')
    
    args = parser.parse_args()
    
    model, train_loader, val_loader, test_loader = train_speech_recognition(overfit_mode=args.overfit)