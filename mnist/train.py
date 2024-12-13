import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mnist import load_mnist
from model import MLP

import time


def accuracy(model, inputs, targets):
    pred = model(inputs).argmax(axis=1)
    return mx.mean(pred == targets)


def eval_step(model, inputs, targets):
    return accuracy(model, inputs, targets)


def loss_fn(model, inputs, targets):
    logits = model(inputs)
    return mx.mean(nn.losses.cross_entropy(logits, targets))


def train_step(model, inputs, targets, optimizer):
    loss, grads = mx.value_and_grad(loss_fn)(model, inputs, targets)
    print(f"Loss: {loss.item():.4f}")
    optimizer.update(model, grads)
    return loss


def main():
    # Hyperparameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Load MNIST dataset
    (train_images, train_labels, test_images, test_labels) = load_mnist()

    # Normalize images to [0, 1] range
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # Convert to MLX arrays
    train_images = mx.array(train_images)
    train_labels = mx.array(train_labels)
    test_images = mx.array(test_images)
    test_labels = mx.array(test_labels)

    # Initialize model and optimizer
    model = MLP()
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Training loop
    num_batches = len(train_images) // batch_size

    print(f"Number of batches: {num_batches}\nBatch size: {batch_size}\n")
    for epoch in range(num_epochs):
        total_loss = 0.0
        start_time = time.time()

        # Shuffle the training data
        perm = mx.random.permutation(len(train_images))
        train_images = train_images[perm]
        train_labels = train_labels[perm]

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_images = train_images[start_idx:end_idx]
            batch_labels = train_labels[start_idx:end_idx]

            loss = train_step(model, batch_images, batch_labels, optimizer)
            mx.eval(loss)
            total_loss += loss.item()

        # Compute training accuracy
        train_accuracy = accuracy(model, train_images, train_labels)
        # Compute test accuracy
        test_accuracy = accuracy(model, test_images, test_labels)

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {total_loss / num_batches:.4f}, "
            f"Train Accuracy: {train_accuracy.item():.4f}, "
            f"Test Accuracy: {test_accuracy.item():.4f}, "
            f"Time: {epoch_time:.2f}s"
        )


if __name__ == "__main__":
    main()
