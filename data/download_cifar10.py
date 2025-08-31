import os
import tarfile
import urllib.request
from tqdm import tqdm


def download_with_progress(url: str, filename: str):
    """Download file with tqdm progress bar"""
    try:
        response = urllib.request.urlopen(url)
        total_size = int(response.headers.get("Content-Length", 0))

        with open(filename, "wb") as file:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=os.path.basename(filename),
            ) as pbar:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    file.write(chunk)
                    pbar.update(len(chunk))
    except Exception as e:
        print(f"Error downloading file: {e}")
        raise


def download_cifar10(data_dir: str = "./cifar-10"):
    """Download and extract CIFAR-10 dataset"""
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = os.path.join(data_dir, "cifar-10-python.tar.gz")

    if not os.path.exists(filename):
        print(f"Downloading CIFAR-10 dataset from {url}")
        download_with_progress(url, filename)

        print(f"Extracting CIFAR-10 dataset to {data_dir}...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(data_dir)
    else:
        print(f"CIFAR-10 foler already exists at {filename}")

    print("Dataset downloaded and extracted")
    return data_dir


if __name__ == "__main__":
    # Example usage
    download_cifar10()
