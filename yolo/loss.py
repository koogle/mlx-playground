"""
YOLOv2 Loss Function Implementation

This module implements the YOLOv2 loss function with anchor boxes, including:
- IoU (Intersection over Union) calculation
- Coordinate prediction loss (x, y, w, h)
- Object confidence loss
- No-object confidence loss
- Class prediction loss

Reference: YOLO9000: Better, Faster, Stronger (https://arxiv.org/abs/1612.08242)
"""

import mlx.core as mx
import math


def compute_box_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two sets of bounding boxes.
    
    Args:
        box1: Predicted boxes [batch, S, S, B, 4] or [batch, S, S, 4]
        box2: Target boxes [batch, S, S, B, 4] or [batch, S, S, 4]
        Both boxes should be in format: [x_center, y_center, width, height]
        
    Returns:
        IoU scores with shape matching the input broadcast shape
    """
    # Convert from (x_center, y_center, width, height) to (x1, y1, x2, y2)
    box1_xy = box1[..., :2]
    box1_wh = box1[..., 2:4]
    box1_wh = mx.maximum(box1_wh, 1e-6)  # Prevent negative or zero width/height
    box1_mins = box1_xy - box1_wh / 2.0  # (x1, y1)
    box1_maxs = box1_xy + box1_wh / 2.0  # (x2, y2)
    
    box2_xy = box2[..., :2]
    box2_wh = box2[..., 2:4]
    box2_wh = mx.maximum(box2_wh, 1e-6)  # Prevent negative or zero width/height
    box2_mins = box2_xy - box2_wh / 2.0
    box2_maxs = box2_xy + box2_wh / 2.0
    
    # Calculate intersection area
    intersect_mins = mx.maximum(box1_mins, box2_mins)
    intersect_maxs = mx.minimum(box1_maxs, box2_maxs)
    intersect_wh = mx.maximum(0.0, intersect_maxs - intersect_mins)
    intersection = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    # Calculate union area
    box1_area = box1_wh[..., 0] * box1_wh[..., 1]
    box2_area = box2_wh[..., 0] * box2_wh[..., 1]
    union = box1_area + box2_area - intersection
    
    # Add small epsilon to avoid division by zero
    iou = intersection / (union + 1e-6)
    return mx.clip(iou, 0.0, 1.0)  # Ensure IoU is between 0 and 1


def validate_inputs(predictions, targets, model):
    """
    Validate input shapes and values for the YOLO loss function.
    
    Args:
        predictions: Model predictions [batch, S, S, B*(5 + C)]
        targets: Ground truth targets [batch, S, S, B*(5 + C)]
        model: YOLO model instance
    
    Raises:
        ValueError: If inputs are invalid
    """
    if len(predictions.shape) != 4:
        raise ValueError(f"Predictions must have 4 dimensions, got {len(predictions.shape)}")
    
    if len(targets.shape) != 4:
        raise ValueError(f"Targets must have 4 dimensions, got {len(targets.shape)}")
    
    batch_size, S, _, channels = predictions.shape
    expected_channels = model.B * (5 + model.C)
    
    if channels != expected_channels:
        raise ValueError(
            f"Predictions should have {expected_channels} channels, got {channels}. "
            f"Check model.B ({model.B}) and model.C ({model.C})"
        )
    
    if targets.shape != predictions.shape:
        raise ValueError(
            f"Targets shape mismatch. Expected {predictions.shape}, got {targets.shape}"
        )


def yolo_loss(predictions, targets, model, lambda_coord=5.0, lambda_noobj=0.5):
    """
    Compute YOLOv2 loss with anchor boxes.
    
    The loss consists of:
    1. Coordinate loss (x, y, w, h) for boxes with objects
    2. Confidence loss for boxes with objects
    3. Confidence loss for boxes without objects
    4. Class prediction loss for boxes with objects
    
    Args:
        predictions: Model predictions [batch, S, S, B*(5 + C)]
        targets: Ground truth targets [batch, S, S, B*(5 + C)]
        model: YOLO model instance
        lambda_coord: Weight for coordinate predictions (default: 5.0)
        lambda_noobj: Weight for no-object confidence predictions (default: 0.5)
        
    Returns:
        Total loss (scalar), averaged over batch
    """
    # Validate inputs
    validate_inputs(predictions, targets, model)
    
    # Extract dimensions
    batch_size = predictions.shape[0]
    S = predictions.shape[1]  # Grid size
    B = model.B  # Number of boxes per cell
    C = model.C  # Number of classes
    
    # Reshape predictions and targets to [batch, S, S, B, 5 + C]
    pred = mx.reshape(predictions, (-1, S, S, B, 5 + C))
    targ = mx.reshape(targets, (-1, S, S, B, 5 + C))
    
    # Split predictions into components
    pred_xy = mx.sigmoid(pred[..., 0:2])  # Center coordinates [batch, S, S, B, 2]
    pred_wh = pred[..., 2:4]  # Width/height (raw)
    pred_conf = mx.sigmoid(pred[..., 4:5])  # Object confidence [batch, S, S, B, 1]
    pred_class = mx.softmax(pred[..., 5:], axis=-1)  # Class probabilities [batch, S, S, B, C]
    
    # Split targets into components
    targ_xy = targ[..., 0:2]  # Center coordinates [batch, S, S, B, 2]
    targ_wh = targ[..., 2:4]  # Width/height
    targ_conf = targ[..., 4:5]  # Object confidence [batch, S, S, B, 1]
    targ_class = targ[..., 5:]  # Class probabilities [batch, S, S, B, C]
    
    # Create grid offsets [1, S, S, 1, 2]
    grid_x, grid_y = mx.meshgrid(mx.arange(S, dtype=mx.float32), 
                                mx.arange(S, dtype=mx.float32))
    grid = mx.stack([grid_x, grid_y], axis=-1)  # [S, S, 2]
    grid = mx.expand_dims(mx.expand_dims(grid, axis=0), axis=3)  # [1, S, S, 1, 2]
    
    # Convert predictions to absolute coordinates
    pred_xy_abs = (pred_xy + grid) / S  # Add grid offsets and normalize
    anchors = mx.reshape(model.anchors, (1, 1, 1, B, 2))  # [1, 1, 1, B, 2]
    pred_wh_abs = mx.exp(pred_wh) * anchors  # Scale by anchors
    pred_boxes = mx.concatenate([pred_xy_abs, pred_wh_abs], axis=-1)
    
    # Convert targets to absolute coordinates
    targ_xy_abs = (targ_xy + grid) / S
    targ_wh_abs = targ_wh * anchors
    targ_boxes = mx.concatenate([targ_xy_abs, targ_wh_abs], axis=-1)
    
    # Object mask (1 for objects, 0 for no objects)
    obj_mask = targ_conf
    noobj_mask = 1.0 - obj_mask
    
    # Count number of objects for normalization
    num_objects = mx.sum(obj_mask) + 1e-6
    
    # 1. Coordinate loss (only for cells with objects)
    # XY loss (using grid-relative coordinates)
    xy_loss = mx.sum(
        obj_mask * ((pred_xy - targ_xy) ** 2)
    ) / num_objects
    
    # WH loss (using anchor-relative coordinates)
    wh_loss = mx.sum(
        obj_mask * ((pred_wh - mx.log(targ_wh + 1e-6)) ** 2)
    ) / num_objects
    
    # 2. Object confidence loss (for cells with objects)
    # Calculate IoU between predicted and target boxes
    ious = compute_box_iou(pred_boxes, targ_boxes)  # [batch, S, S, B]
    obj_loss = mx.sum(
        obj_mask * (pred_conf - mx.expand_dims(ious, axis=-1)) ** 2  # Add extra dim to match shape
    ) / num_objects
    
    # 3. No-object confidence loss (for cells without objects)
    noobj_loss = mx.sum(
        noobj_mask * pred_conf ** 2
    ) / (mx.sum(noobj_mask) + 1e-6)
    
    # 4. Class prediction loss (only for cells with objects)
    class_loss = mx.sum(
        obj_mask * mx.sum((pred_class - targ_class) ** 2, axis=-1, keepdims=True)
    ) / num_objects
    
    # Combine all losses with their respective weights
    total_loss = (
        lambda_coord * (xy_loss + wh_loss) +  # Coordinate loss
        obj_loss +                            # Object confidence loss
        lambda_noobj * noobj_loss +           # No-object confidence loss
        class_loss                            # Class prediction loss
    )
    
    return total_loss
