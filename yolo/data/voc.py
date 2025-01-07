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
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate IoU between two boxes in center format (x, y, w, h)"""
    # Convert to corner format
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2

    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2

    # Calculate intersection area
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)

    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    # Calculate union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    return inter_area / (
        union_area + 1e-6
    )  # Add small epsilon to avoid division by zero


def augment_image(
    image: np.ndarray, boxes: np.ndarray, target_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply data augmentation to image and boxes"""
    # TODO: Implement data augmentation
    # For now, just resize the image and adjust boxes accordingly
    return image, boxes


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
        self.num_classes = len(VOC_CLASSES)
        self.augment = augment and image_set == "train"  # Only augment training data

        # Anchor boxes (precomputed using k-means clustering on training set)
        self.anchors = np.array(
            [
                [1.3221, 1.73145],
                [3.19275, 4.00944],
            ]
        )

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
        """Load and preprocess image"""
        # Get image path
        image_path = os.path.join(self.image_dir, self.image_ids[idx] + ".jpg")
        image = Image.open(image_path).convert("RGB")

        # Resize
        image = image.resize((self.img_size, self.img_size))

        # Convert to numpy and preprocess
        image = np.array(image, dtype=np.float32)
        image = self.preprocess_image(image)

        if self.augment:
            anno = self._get_annotation(idx)
            image, anno["boxes"] = augment_image(image, anno["boxes"], self.img_size)

        return image

    def _convert_to_grid(self, boxes, classes):
        """Convert boxes to grid format"""
        grid_size = self.grid_size
        target = np.zeros((grid_size, grid_size, 5), dtype=np.float32)

        for box, cls in zip(boxes, classes):
            # Get grid cell location
            center_x, center_y = box[:2]
            grid_x = int(center_x * grid_size)
            grid_y = int(center_y * grid_size)

            # Convert coordinates relative to grid cell
            x = center_x * grid_size - grid_x
            y = center_y * grid_size - grid_y
            w = box[2]
            h = box[3]

            # Store in target array (only one best box per cell)
            if target[grid_y, grid_x, 4] == 0:  # If no object already there
                target[grid_y, grid_x] = [x, y, w, h, 1]

        return target

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[mx.array, mx.array]:
        """Get a single training example"""
        image_id = self.image_ids[idx]

        # Load image and annotations
        image = self._load_image(idx)
        annotations = self._get_annotation(idx)

        boxes = annotations["boxes"]
        labels = annotations["classes"]

        # Convert to target format
        target = self.convert_to_target(boxes, labels, image.shape[:2])

        # Convert to MLX arrays
        image = mx.array(image)
        target = mx.array(target)

        return image, target

    def convert_to_target(self, boxes, labels, image_shape):
        """Convert boxes and labels to YOLO target format"""
        S = self.grid_size
        C = 20  # VOC has 20 classes

        # Initialize target grid
        target = np.zeros((S, S, 5 + C))  # x,y,w,h,obj + class_probs

        # Convert boxes to relative coordinates
        height, width = image_shape
        boxes = boxes.copy()
        boxes[:, [0, 2]] /= width  # x coordinates
        boxes[:, [1, 3]] /= height  # y coordinates

        # Convert from corners to center format
        boxes_ctr = np.zeros_like(boxes)
        boxes_ctr[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # cx
        boxes_ctr[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # cy
        boxes_ctr[:, 2] = boxes[:, 2] - boxes[:, 0]  # w
        boxes_ctr[:, 3] = boxes[:, 3] - boxes[:, 1]  # h

        for box, label in zip(boxes_ctr, labels):
            cx, cy, w, h = box

            # Get grid cell indices
            grid_x = int(S * cx)
            grid_y = int(S * cy)

            # Handle edge cases
            grid_x = min(grid_x, S - 1)
            grid_y = min(grid_y, S - 1)

            # Convert coordinates to be relative to grid cell
            cx_rel = S * cx - grid_x
            cy_rel = S * cy - grid_y

            # Set target values
            if target[grid_y, grid_x, 4] == 0:  # If no object already assigned
                target[grid_y, grid_x, 0] = cx_rel
                target[grid_y, grid_x, 1] = cy_rel
                target[grid_y, grid_x, 2] = w
                target[grid_y, grid_x, 3] = h
                target[grid_y, grid_x, 4] = 1  # Object confidence
                target[grid_y, grid_x, 5 + label] = 1  # Class one-hot encoding

        return target

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Image should already be resized and normalized from _load_image
        # Just need to ensure it's in the right format
        if len(image.shape) == 2:
            # Add channel dimension for grayscale images
            image = np.expand_dims(image, axis=-1)

        # Ensure we have 3 channels
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] != 3:
            raise ValueError(f"Expected 3 channels, got {image.shape[-1]}")

        # Ensure float32 type and [0,1] range
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0

        return image


class DataLoader:
    def __init__(self, dataset: VOCDataset, batch_size: int, shuffle: bool = True):
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

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.batch_idx = 0
        return self

    def __next__(self):
        if self.batch_idx >= self.num_batches:
            raise StopIteration

        # Get indices for current batch
        start_idx = self.batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]

        # Process batch
        batch_images = []
        batch_targets = []

        for idx in batch_indices:
            image, target = self.dataset[idx]
            batch_images.append(image)
            batch_targets.append(target)

        # Stack and convert to MLX arrays
        images = mx.stack(batch_images)
        targets = mx.stack(batch_targets)
        mx.eval(images)
        mx.eval(targets)

        self.batch_idx += 1
        return images, targets


def create_data_loader(
    dataset: VOCDataset, batch_size: int, shuffle: bool = True
) -> DataLoader:
    """Create batched data loader"""
    return DataLoader(dataset, batch_size, shuffle)
