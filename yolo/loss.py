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
        predictions: Model predictions
        targets: Ground truth targets
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
    
    if targets.shape[1:] != (S, S, 5 + model.C):
        raise ValueError(
            f"Targets shape mismatch. Expected {(batch_size, S, S, 5 + model.C)}, "
            f"got {targets.shape}"
        )


def yolo_loss(predictions, targets, model, lambda_coord=2.0, lambda_noobj=0.1):
    """
    Compute YOLOv2 loss with anchor boxes.
    
    The loss consists of:
    1. Coordinate loss (x, y, w, h) for boxes with objects
    2. Confidence loss for boxes with objects
    3. Confidence loss for boxes without objects
    4. Class prediction loss for boxes with objects
    
    Args:
        predictions: Model predictions [batch, S, S, B*(5 + C)]
        targets: Ground truth [batch, S, S, 5 + C]
        model: YOLO model instance (needed for anchor boxes)
        lambda_coord: Weight for coordinate predictions (default: 2.0)
        lambda_noobj: Weight for no-object confidence predictions (default: 0.1)
        
    Returns:
        Total loss (scalar), averaged over batch
    """
    # Validate inputs
    validate_inputs(predictions, targets, model)
    
    # Extract dimensions
    batch_size = predictions.shape[0]
    S = predictions.shape[1]  # Grid size
    B = model.B  # Number of anchor boxes per cell
    C = model.C  # Number of classes
    
    # Reshape predictions to [batch, S, S, B, 5 + C]
    pred = mx.reshape(predictions, (-1, S, S, B, 5 + C))
    
    # Split predictions into components
    pred_xy = mx.sigmoid(pred[..., 0:2])  # Center coordinates [batch, S, S, B, 2]
    pred_wh = mx.exp(mx.clip(pred[..., 2:4], -math.log(1e4), math.log(1e4)))  # Width/height
    pred_conf = mx.sigmoid(pred[..., 4:5])  # Object confidence [batch, S, S, B, 1]
    pred_class = mx.sigmoid(pred[..., 5:])  # Class probabilities [batch, S, S, B, C]
    
    # Process targets
    target_boxes = mx.expand_dims(targets[..., :4], axis=3)    # [batch, S, S, 1, 4]
    target_obj = mx.reshape(targets[..., 4:5], (batch_size, S, S, 1, 1))  # [batch, S, S, 1, 1]
    target_class = targets[..., 5:]  # [batch, S, S, C]
    
    # Create grid offsets [1, S, S, 1, 2]
    grid_x, grid_y = mx.meshgrid(mx.arange(S, dtype=mx.float32), 
                                mx.arange(S, dtype=mx.float32))
    grid = mx.stack([grid_x, grid_y], axis=-1)  # [S, S, 2]
    grid = mx.expand_dims(mx.expand_dims(grid, axis=0), axis=3)  # [1, S, S, 1, 2]
    
    # Convert predictions to absolute coordinates
    pred_xy_abs = (pred_xy + grid) / S  # Add grid offsets and normalize
    anchors = mx.reshape(model.anchors, (1, 1, 1, B, 2))  # [1, 1, 1, B, 2]
    pred_wh_abs = pred_wh * anchors  # Scale by anchors
    pred_boxes_abs = mx.concatenate([pred_xy_abs, pred_wh_abs], axis=-1)
    
    # Find responsible boxes using IoU
    ious = mx.stack([
        compute_box_iou(pred_boxes_abs[..., i, :], target_boxes[..., 0, :])
        for i in range(B)
    ], axis=-1)  # [batch, S, S, B]
    
    best_ious = mx.max(ious, axis=-1, keepdims=True)  # [batch, S, S, 1]
    responsible_mask = mx.expand_dims(ious >= best_ious, axis=-1)  # [batch, S, S, B, 1]
    responsible_mask = responsible_mask * target_obj  # Only for cells with objects
    
    # Count number of responsible boxes for normalization
    num_responsible = mx.sum(responsible_mask) + 1e-6
    
    # 1. Coordinate loss (only for responsible boxes)
    # XY loss (normalized by grid size)
    xy_loss = mx.sum(
        responsible_mask *
        (pred_xy - (target_boxes[..., :2] * S - grid)) ** 2
    ) / (num_responsible * S)
    
    # WH loss (normalized and using sqrt)
    wh_loss = mx.sum(
        responsible_mask *
        (mx.sqrt(pred_wh + 1e-6) - mx.sqrt(target_boxes[..., 2:4] / anchors + 1e-6)) ** 2
    ) / num_responsible
    
    # 2. Object confidence loss (for responsible boxes)
    obj_loss = mx.sum(
        responsible_mask * (pred_conf - ious[..., None]) ** 2
    ) / num_responsible
    
    # 3. No-object confidence loss (for non-responsible boxes)
    noobj_mask = (1 - responsible_mask) * (1 - target_obj)
    num_noobj = mx.sum(noobj_mask) + 1e-6
    noobj_loss = mx.sum(noobj_mask * pred_conf ** 2) / num_noobj
    
    # 4. Class prediction loss (only for cells with objects and responsible boxes)
    class_loss = mx.sum(
        responsible_mask * 
        (pred_class - mx.expand_dims(target_class, axis=3)) ** 2
    ) / num_responsible
    
    # Combine all losses with their respective weights
    total_loss = (
        lambda_coord * (xy_loss + wh_loss) +  # Coordinate loss
        2.0 * obj_loss +                      # Boost object confidence loss
        lambda_noobj * noobj_loss +           # No-object confidence loss
        class_loss                            # Class prediction loss
    )
    
    return total_loss
