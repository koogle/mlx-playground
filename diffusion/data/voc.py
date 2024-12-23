import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import mlx.core as mx
from collections import Counter

# PASCAL VOC class names
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
]

def create_description(objects: List[str]) -> str:
    """Create a natural language description from object labels"""
    # Count occurrences of each object
    counter = Counter(objects)
    
    # Create description parts
    parts = []
    for obj, count in counter.items():
        if count > 1:
            parts.append(f"{count} {obj}s")
        else:
            parts.append(obj)
    
    # Join parts with proper grammar
    if len(parts) == 0:
        return "an empty scene"
    elif len(parts) == 1:
        return f"a scene with {parts[0]}"
    elif len(parts) == 2:
        return f"a scene with {parts[0]} and {parts[1]}"
    else:
        return f"a scene with {', '.join(parts[:-1])}, and {parts[-1]}"


class VOCDiffusionDataset:
    """
    PASCAL VOC Dataset loader for text-conditioned diffusion models
    """
    def __init__(
        self,
        data_dir: str,
        year: str = "2012",
        image_set: str = "train",
        img_size: int = 64,
        augment: bool = True,
    ):
        """
        Args:
            data_dir: Root directory of VOC dataset
            year: Dataset year ('2007' or '2012')
            image_set: 'train', 'val', or 'test'
            img_size: Input image size
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.year = year
        self.image_set = image_set
        self.img_size = img_size
        self.augment = augment and image_set == "train"

        # Paths
        self.image_dir = os.path.join(data_dir, "JPEGImages")
        self.annotation_dir = os.path.join(data_dir, "Annotations")
        self.image_set_file = os.path.join(
            data_dir, "ImageSets", "Main", f"{image_set}.txt"
        )

        # Load image IDs
        self.image_ids = self._load_image_ids()

    def _load_image_ids(self) -> List[str]:
        """Load image IDs from dataset file"""
        with open(self.image_set_file) as f:
            return [line.strip() for line in f.readlines()]

    def _get_annotation(self, idx: int) -> Tuple[List[str], List[List[float]]]:
        """Load annotation for a given index"""
        img_id = self.image_ids[idx]
        annotation_path = os.path.join(self.annotation_dir, f"{img_id}.xml")
        
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # Extract object names and boxes
        objects = []
        boxes = []
        
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name in VOC_CLASSES:
                objects.append(name)
                
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                boxes.append([xmin, ymin, xmax, ymax])
        
        return objects, boxes

    def _load_image(self, idx: int) -> np.ndarray:
        """Load and preprocess image"""
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Resize
        image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        
        # Convert to numpy array and normalize
        image = np.array(image, dtype=np.float32) / 255.0
        
        # Convert to CHW format
        image = np.transpose(image, (2, 0, 1))
        
        return image

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        """Get a single sample"""
        # Load image
        image = self._load_image(idx)
        
        # Load annotation and create description
        objects, _ = self._get_annotation(idx)
        description = create_description(objects)
        
        return image, description


class DataLoader:
    """
    DataLoader for the VOC dataset that returns batches of (image, description) pairs
    """
    def __init__(self, dataset: VOCDiffusionDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Calculate number of batches
        self.num_batches = len(dataset) // batch_size
        if len(dataset) % batch_size != 0:
            self.num_batches += 1
        
        self.batch_idx = 0
    
    def __len__(self) -> int:
        return self.num_batches
    
    def __iter__(self):
        self.batch_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self) -> Tuple[mx.array, List[str]]:
        if self.batch_idx >= self.num_batches:
            raise StopIteration
        
        start_idx = self.batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.dataset))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Get batch data
        images = []
        descriptions = []
        for idx in batch_indices:
            image, desc = self.dataset[idx]
            images.append(image)
            descriptions.append(desc)
        
        # Stack images into a batch
        images = mx.array(np.stack(images))
        
        self.batch_idx += 1
        return images, descriptions


def create_data_loader(
    dataset: VOCDiffusionDataset, batch_size: int, shuffle: bool = True
) -> DataLoader:
    """Create a batched data loader"""
    return DataLoader(dataset, batch_size, shuffle)
