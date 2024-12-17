import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import mlx.core as mx

# PASCAL VOC class names
VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# Create class to index mapping
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(VOC_CLASSES)}


def augment_image(
    image: np.ndarray, boxes: np.ndarray, target_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply data augmentation to image and boxes"""
    # Convert any float type [0-1] to uint8 [0-255]
    if np.issubdtype(image.dtype, np.floating):
        image = (image * 255.0).clip(0, 255).astype(np.uint8)

    # Random horizontal flip
    if np.random.random() < 0.5:
        image = np.fliplr(image).astype(np.uint8)
        boxes = boxes.copy()
        boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]  # Flip x coordinates

    # Random scaling (zoom in/out)
    scale = np.float32(np.random.uniform(0.8, 1.2))
    h, w = image.shape[:2]
    nh, nw = int(h * scale), int(w * scale)
    image = np.array(Image.fromarray(image).resize((nw, nh)), dtype=np.uint8)

    # Adjust box coordinates for scaling
    boxes = boxes.copy()
    boxes[:, [0, 2]] *= np.float32(nw) / w
    boxes[:, [1, 3]] *= np.float32(nh) / h

    # Random brightness
    if np.random.random() < 0.5:
        delta = np.float32(np.random.uniform(-32, 32))
        image = np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)

    # Random saturation
    if np.random.random() < 0.5:
        saturation = np.float32(np.random.uniform(0.5, 1.5))
        hsv = np.array(Image.fromarray(image).convert("HSV"), dtype=np.uint8)
        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1].astype(np.float32) * saturation, 0, 255
        ).astype(np.uint8)
        image = np.array(
            Image.fromarray(hsv, mode="HSV").convert("RGB"), dtype=np.uint8
        )

    # Resize back to target size
    if image.shape[:2] != (target_size, target_size):
        # Adjust box coordinates for final resize
        curr_h, curr_w = image.shape[:2]
        boxes[:, [0, 2]] *= np.float32(target_size) / curr_w
        boxes[:, [1, 3]] *= np.float32(target_size) / curr_h
        image = np.array(
            Image.fromarray(image).resize((target_size, target_size)), dtype=np.uint8
        )

    # Convert back to float32 [0-1]
    return image.astype(np.float32) / 255.0, boxes


class VOCDataset:
    def __init__(
        self,
        data_dir: str,
        year: str = "2012",
        image_set: str = "train",
        img_size: int = 448,
        grid_size: int = 7,
        num_boxes: int = 2,
        augment: bool = True,
    ):
        """
        PASCAL VOC Dataset loader

        Args:
            data_dir: Root directory of VOC dataset
            year: Dataset year ('2007' or '2012')
            image_set: 'train', 'val', or 'test'
            img_size: Input image size (448 for YOLO)
            grid_size: Grid size (7 for YOLO)
            num_boxes: Number of boxes per grid cell (2 for YOLO)
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.year = year
        self.image_set = image_set
        self.img_size = img_size
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.augment = augment and image_set == "train"  # Only augment training data

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

    def _get_annotation(self, idx: int) -> Dict:
        """Load annotation for a given index"""
        img_id = self.image_ids[idx]
        anno_path = os.path.join(self.annotation_dir, f"{img_id}.xml")

        # Parse XML
        tree = ET.parse(anno_path)
        root = tree.getroot()

        # Image size
        size = root.find("size")

        width = float(size.find("width").text)
        height = float(size.find("height").text)

        # Boxes and classes
        boxes = []
        classes = []

        for obj in root.iter("object"):
            # Get class
            class_name = obj.find("name").text
            if class_name not in CLASS_TO_IDX:
                continue

            # Get bbox
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text) / width
            ymin = float(bbox.find("ymin").text) / height
            xmax = float(bbox.find("xmax").text) / width
            ymax = float(bbox.find("ymax").text) / height

            # Convert to YOLO format (center_x, center_y, w, h)
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin

            boxes.append([center_x, center_y, w, h])
            classes.append(CLASS_TO_IDX[class_name])

        return {
            "boxes": np.array(boxes, dtype=np.float32),
            "classes": np.array(classes, dtype=np.int64),
            "image_id": img_id,
        }

    def _load_image(self, idx: int) -> np.ndarray:
        """Load image from disk and preprocess"""
        # Get image path
        image_path = os.path.join(self.image_dir, self.image_ids[idx] + ".jpg")
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.img_size, self.img_size))

        # Convert to numpy array and normalize
        image = np.array(image, dtype=np.float32) / 255.0

        if self.augment:
            anno = self._get_annotation(idx)
            image, anno["boxes"] = augment_image(image, anno["boxes"], self.img_size)

        return image.astype(np.float32)  # Ensure float32 type

    def _convert_to_grid(self, boxes: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """Convert boxes and classes to YOLO grid format"""
        grid_size = self.grid_size
        num_classes = len(VOC_CLASSES)

        # Initialize target grid - shape: [S, S, 5 + C] where 5 is [x, y, w, h, confidence]
        target = np.zeros((grid_size, grid_size, 5 + num_classes), dtype=np.float32)

        for box, cls in zip(boxes, classes):
            # Get grid cell location
            center_x, center_y = box[:2]
            grid_x = int(center_x * grid_size)
            grid_y = int(center_y * grid_size)

            # Constrain to grid bounds
            grid_x = min(grid_size - 1, max(0, grid_x))
            grid_y = min(grid_size - 1, max(0, grid_y))

            # Convert box coordinates relative to grid cell
            x = center_x * grid_size - grid_x
            y = center_y * grid_size - grid_y
            w = box[2]
            h = box[3]

            # Set box coordinates and confidence
            target[grid_y, grid_x, :4] = [x, y, w, h]
            target[grid_y, grid_x, 4] = 1  # Object confidence

            # Set class probability
            target[grid_y, grid_x, 5 + cls] = 1

        return target

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[mx.array, mx.array]:
        """Get a single sample"""
        # Load image and annotations
        image = self._load_image(idx)
        anno = self._get_annotation(idx)

        # Convert to grid format
        target = self._convert_to_grid(anno["boxes"], anno["classes"])

        # Convert to MLX arrays - keep channels last (NHWC format)
        image = mx.array(image)  # Shape: (H, W, C)
        target = mx.array(target)

        return image, target


def create_data_loader(
    dataset: VOCDataset, batch_size: int, shuffle: bool = True
) -> Tuple[List[mx.array], List[mx.array]]:
    """Create batched data loader"""
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)

    # Create batches with minimal memory overhead
    images = []
    targets = []
    batch_indices = []

    for idx in indices:
        batch_indices.append(idx)

        if len(batch_indices) == batch_size:
            # Process batch
            batch_images = []
            batch_targets = []

            for bidx in batch_indices:
                image, target = dataset[bidx]
                batch_images.append(image)
                batch_targets.append(target)

            # Stack and convert to MLX arrays
            images.append(mx.stack(batch_images))
            targets.append(mx.stack(batch_targets))

            # Clear batch indices
            batch_indices = []
            mx.eval(images[-1])  # Force evaluation to free memory
            mx.eval(targets[-1])

    # Handle remaining samples
    if batch_indices:
        batch_images = []
        batch_targets = []
        for bidx in batch_indices:
            image, target = dataset[bidx]
            batch_images.append(image)
            batch_targets.append(target)
        images.append(mx.stack(batch_images))
        targets.append(mx.stack(batch_targets))
        mx.eval(images[-1])
        mx.eval(targets[-1])

    return images, targets
