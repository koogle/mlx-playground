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
        learning_rate = 1e-3  # Same learning rate to avoid NaN
        num_epochs = 50  # Enough epochs to see clear overfitting
        print("Running in overfit mode - using only 3 samples")
    else:
        batch_size = 32
        learning_rate = 1e-3
        num_epochs = 10
    
    # Model parameters  
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
    
    
    # Optimizer
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
        return nn.losses.cross_entropy(pooled_logits, y)
    
    def evaluate(model, data_loader):
        """Evaluate model accuracy"""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for features, labels in data_loader.create_batches(batch_size=batch_size, shuffle=False):
            # Forward pass
            logits = model(features)
            pooled_logits = mx.mean(logits, axis=1)
            
            # Loss
            loss = nn.losses.cross_entropy(pooled_logits, labels)
            total_loss += loss.item()
            
            # Accuracy
            predicted = mx.argmax(pooled_logits, axis=1)
            correct = mx.sum(predicted == labels).item()
            total_correct += correct
            total_samples += len(labels)
        
        avg_loss = total_loss / (total_samples / batch_size)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy
    
    # Training loop
    print("Starting training...")
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        # Training
        for features, labels in train_loader.create_batches(batch_size=batch_size, shuffle=True):
            # Forward and backward pass
            loss, grads = mx.value_and_grad(loss_fn)(model.parameters(), features, labels)
            
            # Update parameters
            optimizer.update(model, grads)
            mx.eval(model.parameters())
            
            total_loss += loss.item()
            num_batches += 1
            
            # More frequent logging in overfit mode
            log_freq = 1 if overfit_mode else 50
            if num_batches % log_freq == 0:
                print(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {loss.item():.4f}")
        
        # Validation
        val_loss, val_accuracy = evaluate(model, val_loader)
        avg_train_loss = total_loss / num_batches
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
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