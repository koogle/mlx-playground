import os
import numpy as np
import mlx.core as mx
from typing import Tuple, Optional
import pickle
import gzip
from data.download_imagenet64 import download_imagenet64


def unpickle(file):
    """Load ImageNet-64 batch file"""
    if file.endswith('.gz'):
        with gzip.open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
    else:
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
    return dict


def load_batch(batch_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single ImageNet-64 batch file
    
    Returns:
        (images, labels) tuple where:
        - images: (N, 64, 64, 3) array of images in CHW format  
        - labels: (N,) array of labels
    """
    batch = unpickle(batch_path)
    data = batch[b'data']
    labels = batch[b'labels']
    
    # Reshape data: (num_samples, 12288) -> (num_samples, 3, 64, 64)
    # ImageNet-64 has 64x64x3 = 12288 pixels per image
    data = data.reshape(-1, 3, 64, 64)
    
    return data, np.array(labels)


def _ensure_dataset_exists(data_dir: str, download: bool) -> str:
    extracted_dir = os.path.join(data_dir, "imagenet64")
    
    if not os.path.exists(extracted_dir):
        if download:
            download_imagenet64(data_dir)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {extracted_dir}. Set download=True to download it."
            )
    
    return extracted_dir


def load_file(
    file_name: str,
    data_dir: str = "./imagenet64",
    download: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    extracted_dir = _ensure_dataset_exists(data_dir, download)
    
    file_path = os.path.join(extracted_dir, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    images, labels = load_batch(file_path)
    
    if normalize:
        images = images.astype(np.float32) / 255.0
    
    return images, labels


def load_train_data(
    data_dir: str = "./imagenet64", download: bool = True, normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ImageNet-64 training data
    
    Args:
        data_dir: Directory containing the dataset
        download: Whether to download if not found
        normalize: Whether to normalize pixel values to [0, 1]
    
    Returns:
        (train_images, train_labels) tuple where:
        - train_images: (N, 64, 64, 3) array
        - train_labels: (N,) array
    """
    extracted_dir = _ensure_dataset_exists(data_dir, download)
    
    # Find all train batch files (typically train_data_batch_1.npz to train_data_batch_10.npz)
    batch_files = sorted([
        f for f in os.listdir(extracted_dir) 
        if f.startswith("train_data_batch_") and (f.endswith('.npz') or f.endswith('.gz'))
    ])
    
    if not batch_files:
        # Fallback to looking for any train files
        batch_files = sorted([
            f for f in os.listdir(extracted_dir) 
            if 'train' in f and (f.endswith('.npz') or f.endswith('.gz') or f.endswith('.pkl'))
        ])
    
    if not batch_files:
        raise FileNotFoundError(f"No training batch files found in {extracted_dir}")
    
    images_list = []
    labels_list = []
    
    for batch_file in batch_files:
        images, labels = load_file(
            batch_file, data_dir, download=False, normalize=normalize
        )
        images_list.append(images)
        labels_list.append(labels)
    
    # Concatenate all batches
    train_images = np.concatenate(images_list, axis=0)
    train_labels = np.concatenate(labels_list, axis=0)
    
    print(f"Loaded ImageNet-64 training data: {train_images.shape[0]} samples")
    return train_images, train_labels


def load_test_data(
    data_dir: str = "./imagenet64", download: bool = True, normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ImageNet-64 test/validation data
    
    Args:
        data_dir: Directory containing the dataset  
        download: Whether to download if not found
        normalize: Whether to normalize pixel values to [0, 1]
    
    Returns:
        (test_images, test_labels) tuple where:
        - test_images: (N, 64, 64, 3) array
        - test_labels: (N,) array
    """
    extracted_dir = _ensure_dataset_exists(data_dir, download)
    
    # Look for validation/test files
    possible_files = ["val_data.npz", "test_data.npz", "val_data.gz", "test_data.gz"]
    
    test_file = None
    for fname in possible_files:
        if os.path.exists(os.path.join(extracted_dir, fname)):
            test_file = fname
            break
    
    if test_file is None:
        raise FileNotFoundError(f"No test/validation file found in {extracted_dir}")
    
    images, labels = load_file(test_file, data_dir, download, normalize)
    print(f"Loaded ImageNet-64 test data: {images.shape[0]} samples")
    return images, labels


def load_imagenet64(
    data_dir: str = "./imagenet64",
    download: bool = True,
    normalize: bool = True,
    load_test: bool = True,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray]]
]:
    """
    Load complete ImageNet-64 dataset
    
    Args:
        data_dir: Directory to save/load the dataset
        download: Whether to download the dataset if not found
        normalize: Whether to normalize pixel values to [0, 1]
        load_test: Whether to load the test dataset
    
    Returns:
        ((train_images, train_labels), (test_images, test_labels))
        - train_images: (N, 64, 64, 3) array
        - train_labels: (N,) array  
        - test_images: (M, 64, 64, 3) array or None if load_test=False
        - test_labels: (M,) array or None if load_test=False
    """
    # Load training data
    train_images, train_labels = load_train_data(data_dir, download, normalize)
    
    # Load test data if requested
    if load_test:
        try:
            test_images, test_labels = load_test_data(data_dir, download, normalize)
        except FileNotFoundError:
            print("Warning: Test data not found, continuing with training data only")
            test_images, test_labels = None, None
    else:
        test_images, test_labels = None, None
    
    print("ImageNet-64 dataset loaded successfully")
    print(f"  Image shape: {train_images.shape[1:]}")
    print(f"  Num classes: {len(np.unique(train_labels))}")
    
    return (train_images, train_labels), (test_images, test_labels)


class ImageNet64DataLoader:
    """Simple data loader for ImageNet-64 with batching support"""
    
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
        
        # Always convert to MLX arrays
        batch_images = mx.array(batch_images)
        batch_labels = mx.array(batch_labels)
        
        return batch_images, batch_labels
    
    def __len__(self):
        return self.num_batches


if __name__ == "__main__":
    # Example usage - load only training data for diffusion
    try:
        train_images, train_labels = load_train_data()
        print(f"Training data shape: {train_images.shape}")
        
        # Create data loader
        train_loader = ImageNet64DataLoader(
            train_images, train_labels, batch_size=64, shuffle=True
        )
        
        # Example: iterate through one batch
        for batch_images, batch_labels in train_loader:
            print(f"Batch shape: {batch_images.shape}")
            print(f"Labels shape: {batch_labels.shape}")
            print(f"Image range: [{batch_images.min():.3f}, {batch_images.max():.3f}]")
            break
            
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("Please download the ImageNet-64 dataset first")