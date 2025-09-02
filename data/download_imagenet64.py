import os
import urllib.request
import tarfile
import zipfile
from tqdm import tqdm


class DownloadProgressHook:
    """Progress hook for urllib.request.urlretrieve"""
    
    def __init__(self):
        self.pbar = None
    
    def __call__(self, count, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
        
        progress = min(count * block_size, total_size)
        self.pbar.update(progress - self.pbar.n)
        
        if progress >= total_size:
            self.pbar.close()


def download_imagenet64(data_dir: str = "./imagenet64"):
    """
    Download ImageNet-64 dataset
    
    Note: This is a placeholder implementation. The actual ImageNet-64 dataset
    requires registration and access through academic channels. 
    
    For development/testing, you can:
    1. Download from academic sources with proper access
    2. Use a subset for testing
    3. Generate synthetic data for development
    
    Args:
        data_dir: Directory to save the dataset
    """
    os.makedirs(data_dir, exist_ok=True)
    
    print("ImageNet-64 Dataset Download")
    print("=" * 50)
    print("⚠️  IMPORTANT NOTICE:")
    print("ImageNet-64 requires academic access and registration.")
    print("This function serves as a placeholder for the download process.")
    print()
    print("To obtain ImageNet-64:")
    print("1. Register at https://www.image-net.org/")
    print("2. Request access to the downsampled datasets")
    print("3. Download the 64x64 version")
    print("4. Extract to:", os.path.abspath(data_dir))
    print()
    print("Expected structure:")
    print(f"{data_dir}/")
    print("  ├── train_data_batch_1.npz  (or .gz/.pkl)")
    print("  ├── train_data_batch_2.npz")
    print("  ├── ... (more training batches)")
    print("  └── val_data.npz  (validation set)")
    print()
    
    # Check if user has manually placed files
    expected_files = [
        "train_data_batch_1.npz", "train_data_batch_1.gz", "train_data_batch_1.pkl",
        "train_data.npz", "train_data.gz", "train.pkl"
    ]
    
    existing_files = []
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if any(expected in file for expected in ["train", "val", "test"]):
                existing_files.append(file)
    
    if existing_files:
        print("✅ Found existing files:")
        for file in existing_files[:5]:  # Show first 5
            print(f"   - {file}")
        if len(existing_files) > 5:
            print(f"   ... and {len(existing_files) - 5} more files")
        print("\nDataset appears to be already available!")
        return
    
    # For development purposes, create a small synthetic dataset
    print("🔧 Creating synthetic ImageNet-64 data for development...")
    print("   (Replace with real data when available)")
    
    _create_synthetic_imagenet64(data_dir)
    print("✅ Synthetic dataset created for development/testing")


def _create_synthetic_imagenet64(data_dir: str):
    """
    Create a small synthetic ImageNet-64 dataset for development/testing
    """
    import numpy as np
    import pickle
    
    # Create a small synthetic dataset
    num_samples = 1000
    num_classes = 1000  # ImageNet has 1000 classes
    image_size = 64
    
    # Generate random images and labels
    np.random.seed(42)  # For reproducibility
    
    # Create training data
    train_images = np.random.randint(0, 256, (num_samples, 3, image_size, image_size), dtype=np.uint8)
    train_labels = np.random.randint(0, num_classes, num_samples, dtype=np.int32)
    
    # Save as a single batch file
    train_batch = {
        b'data': train_images.reshape(num_samples, -1),
        b'labels': train_labels.tolist()
    }
    
    train_path = os.path.join(data_dir, "train_data_batch_1.pkl")
    with open(train_path, 'wb') as f:
        pickle.dump(train_batch, f)
    
    # Create validation data (smaller)
    val_samples = 200
    val_images = np.random.randint(0, 256, (val_samples, 3, image_size, image_size), dtype=np.uint8)
    val_labels = np.random.randint(0, num_classes, val_samples, dtype=np.int32)
    
    val_batch = {
        b'data': val_images.reshape(val_samples, -1),
        b'labels': val_labels.tolist()
    }
    
    val_path = os.path.join(data_dir, "val_data.pkl")
    with open(val_path, 'wb') as f:
        pickle.dump(val_batch, f)
    
    print(f"   Created {num_samples} training samples")
    print(f"   Created {val_samples} validation samples")
    print(f"   Saved to: {data_dir}")


if __name__ == "__main__":
    download_imagenet64()