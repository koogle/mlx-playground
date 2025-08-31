import os
import pickle
import numpy as np
from typing import Tuple, Optional
from download_cifar10 import download_cifar10


def unpickle(file):
    """Load CIFAR-10 batch file"""
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def load_batch(batch_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single CIFAR-10 batch file
    
    Args:
        batch_path: Path to the batch file
    
    Returns:
        (images, labels) tuple where:
        - images: (N, 32, 32, 3) array of images in HWC format
        - labels: (N,) array of labels
    """
    batch = unpickle(batch_path)
    data = batch[b"data"]
    labels = batch[b"labels"]

    # Reshape data: (num_samples, 3072) -> (num_samples, 3, 32, 32)
    data = data.reshape(-1, 3, 32, 32)
    # Convert from CHW to HWC format
    data = data.transpose(0, 2, 3, 1)

    return data, np.array(labels)


def load_train_data(
    data_dir: str = "./cifar-10",
    download: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CIFAR-10 training data
    
    Args:
        data_dir: Directory containing the dataset
        download: Whether to download if not found
        normalize: Whether to normalize pixel values to [0, 1]
    
    Returns:
        (train_images, train_labels) tuple where:
        - train_images: (50000, 32, 32, 3) array
        - train_labels: (50000,) array
    """
    extracted_dir = os.path.join(data_dir, "cifar-10-batches-py")

    if not os.path.exists(extracted_dir):
        if download:
            download_cifar10(data_dir)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {extracted_dir}. Set download=True to download it."
            )

    # Find all data_batch files
    batch_files = sorted(
        [f for f in os.listdir(extracted_dir) if f.startswith("data_batch_")]
    )

    if not batch_files:
        raise FileNotFoundError(f"No data_batch files found in {extracted_dir}")

    train_images = []
    train_labels = []

    for batch_file in batch_files:
        batch_path = os.path.join(extracted_dir, batch_file)
        images, labels = load_batch(batch_path)
        train_images.append(images)
        train_labels.append(labels)

    # Concatenate all batches
    train_images = np.concatenate(train_images, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    # Normalize if requested
    if normalize:
        train_images = train_images.astype(np.float32) / 255.0

    print(f"Loaded CIFAR-10 training data: {train_images.shape[0]} samples")
    return train_images, train_labels


def load_test_data(
    data_dir: str = "./cifar-10",
    download: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CIFAR-10 test data
    
    Args:
        data_dir: Directory containing the dataset
        download: Whether to download if not found
        normalize: Whether to normalize pixel values to [0, 1]
    
    Returns:
        (test_images, test_labels) tuple where:
        - test_images: (10000, 32, 32, 3) array
        - test_labels: (10000,) array
    """
    extracted_dir = os.path.join(data_dir, "cifar-10-batches-py")

    if not os.path.exists(extracted_dir):
        if download:
            download_cifar10(data_dir)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {extracted_dir}. Set download=True to download it."
            )

    # Load test batch
    test_file = os.path.join(extracted_dir, "test_batch")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test batch not found at {test_file}")
    
    test_images, test_labels = load_batch(test_file)

    # Normalize if requested
    if normalize:
        test_images = test_images.astype(np.float32) / 255.0

    print(f"Loaded CIFAR-10 test data: {test_images.shape[0]} samples")
    return test_images, test_labels


def load_cifar10(
    data_dir: str = "./cifar-10", 
    download: bool = True, 
    normalize: bool = True,
    load_test: bool = True
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Load complete CIFAR-10 dataset
    
    Args:
        data_dir: Directory to save/load the dataset
        download: Whether to download the dataset if not found
        normalize: Whether to normalize pixel values to [0, 1]
        load_test: Whether to load the test dataset
    
    Returns:
        ((train_images, train_labels), (test_images, test_labels))
        - train_images: (50000, 32, 32, 3) array
        - train_labels: (50000,) array
        - test_images: (10000, 32, 32, 3) array or None if load_test=False
        - test_labels: (10000,) array or None if load_test=False
    """
    # Load training data
    train_images, train_labels = load_train_data(data_dir, download, normalize)
    
    # Load test data if requested
    if load_test:
        test_images, test_labels = load_test_data(data_dir, download, normalize)
    else:
        test_images, test_labels = None, None
    
    print("CIFAR-10 dataset loaded successfully")
    print(f"  Image shape: {train_images.shape[1:]}")
    print(f"  Num classes: {len(np.unique(train_labels))}")
    
    return (train_images, train_labels), (test_images, test_labels)


def get_cifar10_labels():
    """Get CIFAR-10 class labels"""
    return [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]


class CIFAR10DataLoader:
    """Simple data loader for CIFAR-10 with batching support"""

    def __init__(self, images, labels, batch_size=32, shuffle=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(images)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        self.reset()

    def reset(self):
        """Reset the data loader for a new epoch"""
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.batch_idx = 0

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.batch_idx >= self.num_batches:
            raise StopIteration

        start_idx = self.batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]

        batch_images = self.images[batch_indices]
        batch_labels = self.labels[batch_indices]

        self.batch_idx += 1
        return batch_images, batch_labels

    def __len__(self):
        return self.num_batches


if __name__ == "__main__":
    # Example usage - load only training data for diffusion
    train_images, train_labels = load_train_data()
    print(f"Training data shape: {train_images.shape}")
    
    # Or load everything
    (train_images, train_labels), (test_images, test_labels) = load_cifar10()
    
    # Create data loaders
    train_loader = CIFAR10DataLoader(
        train_images, train_labels, batch_size=64, shuffle=True
    )
    
    if test_images is not None:
        test_loader = CIFAR10DataLoader(
            test_images, test_labels, batch_size=64, shuffle=False
        )

    # Get class labels
    class_names = get_cifar10_labels()

    # Example: iterate through one batch
    for batch_images, batch_labels in train_loader:
        print(f"Batch shape: {batch_images.shape}")
        print(f"Labels shape: {batch_labels.shape}")
        print(f"First image label: {class_names[batch_labels[0]]}")
        break