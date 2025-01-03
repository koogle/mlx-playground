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
    box1_x1 = box1[0] - box1[2]/2
    box1_y1 = box1[1] - box1[3]/2
    box1_x2 = box1[0] + box1[2]/2
    box1_y2 = box1[1] + box1[3]/2
    
    box2_x1 = box2[0] - box2[2]/2
    box2_y1 = box2[1] - box2[3]/2
    box2_x2 = box2[0] + box2[2]/2
    box2_y2 = box2[1] + box2[3]/2
    
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
    
    return inter_area / (union_area + 1e-6)  # Add small epsilon to avoid division by zero

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
        self.anchors = np.array([
            [1.3221, 1.73145],
            [3.19275, 4.00944],
        ])

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
        num_classes = self.num_classes
        num_boxes = self.num_boxes

        # Initialize target grid with shape [S, S, B * (5 + C)]
        # For each box: [x, y, w, h, confidence] + [class_probs]
        target = np.zeros((grid_size, grid_size, num_boxes * (5 + num_classes)), dtype=np.float32)

        for box, cls in zip(boxes, classes):
            # Get grid cell location (use proper rounding)
            center_x, center_y = box[:2]
            grid_x = int(np.floor(center_x * grid_size))
            grid_y = int(np.floor(center_y * grid_size))

            # Constrain to grid bounds
            grid_x = min(grid_size - 1, max(0, grid_x))
            grid_y = min(grid_size - 1, max(0, grid_y))

            # Convert box coordinates relative to grid cell
            x = center_x * grid_size - grid_x  # Relative x within cell (0-1)
            y = center_y * grid_size - grid_y  # Relative y within cell (0-1)
            w = box[2]  # Width as ratio of image size
            h = box[3]  # Height as ratio of image size

            # Calculate IoU with anchor boxes
            box_wh = np.array([w * grid_size, h * grid_size])  # Scale to absolute size
            ious = np.array([calculate_iou(
                np.array([0.5, 0.5, anchor[0], anchor[1]]),
                np.array([0.5, 0.5, box_wh[0], box_wh[1]])
            ) for anchor in self.anchors])

            # Assign box to the anchor with highest IoU
            best_anchor = np.argmax(ious)
            
            # Calculate offset in target array
            box_offset = best_anchor * (5 + num_classes)
            
            # Only update if no object or current object has higher IoU
            curr_conf = target[grid_y, grid_x, box_offset + 4]
            if curr_conf < 0.5:  # If no confident object already assigned
                # Set box coordinates and confidence
                target[grid_y, grid_x, box_offset:box_offset + 4] = [x, y, w, h]
                target[grid_y, grid_x, box_offset + 4] = 1  # Object confidence

                # Set class probabilities (one-hot encoding)
                class_offset = box_offset + 5
                target[grid_y, grid_x, class_offset:class_offset + num_classes] = 0
                target[grid_y, grid_x, class_offset + cls] = 1

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
        target = mx.array(target)  # Shape: (S, S, B*(5+C))

        return image, target

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
