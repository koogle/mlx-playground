import os
import tarfile
import urllib.request
import sys

def download_with_progress(url: str, filename: str):
    """Download file with progress bar"""
    def download_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        mb_downloaded = downloaded / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        sys.stdout.write(f'\rDownloading: {percent:.1f}% [{mb_downloaded:.1f}/{mb_total:.1f} MB]')
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, filename, reporthook=download_hook)
        print()  # New line after download completes
    except Exception as e:
        print(f"\nError downloading file: {e}")
        raise

def download_cifar10(data_dir: str = "./cifar-10"):
    """Download and extract CIFAR-10 dataset"""
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = os.path.join(data_dir, "cifar-10-python.tar.gz")
    
    # Download if not already present
    if not os.path.exists(filename):
        print(f"Downloading CIFAR-10 dataset from {url}")
        download_with_progress(url, filename)
    else:
        print(f"CIFAR-10 archive already exists at {filename}")
    
    # Extract if not already extracted
    extracted_dir = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(extracted_dir):
        print(f"Extracting CIFAR-10 dataset to {data_dir}...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(data_dir)
        print("Extraction complete!")
    else:
        print(f"CIFAR-10 already extracted at {extracted_dir}")
    
    print("Dataset ready!")
    return extracted_dir

if __name__ == "__main__":
    # Example usage
    download_cifar10()