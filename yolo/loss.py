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


def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    """Compute focal loss for better handling of hard examples"""
    pred = mx.clip(pred, 1e-6, 1.0 - 1e-6)
    ce_loss = -target * mx.log(pred) - (1 - target) * mx.log(1 - pred)
    p_t = target * pred + (1 - target) * (1 - pred)
    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    return alpha_t * ((1 - p_t) ** gamma) * ce_loss


def validate_inputs(predictions, targets, model):
    """
    Validate input shapes and values for the YOLO loss function.
    
    Args:
        predictions: Model predictions [batch, B*(5 + C), S, S] in NCHW format
        targets: Ground truth targets [batch, S, S, B*(5 + C)] in NHWC format
        model: YOLO model instance
    
    Raises:
        ValueError: If inputs are invalid
    """
    if len(predictions.shape) != 4:
        raise ValueError(f"Predictions must have 4 dimensions, got {len(predictions.shape)}")
    
    if len(targets.shape) != 4:
        raise ValueError(f"Targets must have 4 dimensions, got {len(targets.shape)}")
    
    batch_size, _, S, channels = predictions.shape
    expected_channels = model.B * (5 + model.C)
    
    if channels != expected_channels:
        raise ValueError(
            f"Predictions should have {expected_channels} channels, got {channels}. "
            f"Check model.B ({model.B}) and model.C ({model.C})"
        )
    
    if targets.shape != (batch_size, S, S, expected_channels):
        raise ValueError(
            f"Targets shape mismatch. Expected {(batch_size, S, S, expected_channels)}, got {targets.shape}"
        )


def yolo_loss(predictions, targets, model, lambda_coord=5.0, lambda_noobj=0.5, class_weights=None):
    """
    Compute YOLOv2 loss with anchor boxes and numerical stability improvements.
    """
    # Validate inputs
    validate_inputs(predictions, targets, model)
    
    # Extract dimensions
    batch_size = predictions.shape[0]
    S = model.S  # Grid size
    B = model.B  # Number of boxes per cell
    C = model.C  # Number of classes
    
    # Convert predictions from NCHW to NHWC format
    predictions = mx.transpose(predictions, (0, 2, 3, 1))  # [batch, S, S, B*(5+C)]
    
    # Reshape predictions and targets to [batch, S, S, B, 5 + C]
    predictions = mx.reshape(predictions, (batch_size, S, S, B, 5 + C))
    targets = mx.reshape(targets, (batch_size, S, S, B, 5 + C))
    
    # Extract components from predictions with numerical stability
    pred_xy = mx.sigmoid(mx.clip(predictions[..., 0:2], -10, 10))  # Center coordinates
    pred_wh = mx.clip(predictions[..., 2:4], -10, 10)  # Width/height
    pred_conf = mx.sigmoid(mx.clip(predictions[..., 4:5], -10, 10))  # Object confidence
    pred_class = predictions[..., 5:]  # Class predictions (raw logits)
    
    # Apply softmax to class predictions with numerical stability
    pred_class = pred_class - mx.max(pred_class, axis=-1, keepdims=True)  # For numerical stability
    pred_class = mx.exp(pred_class)
    pred_class = pred_class / (mx.sum(pred_class, axis=-1, keepdims=True) + 1e-10)
    
    # Extract components from targets
    targ_xy = targets[..., 0:2]  # Center coordinates
    targ_wh = targets[..., 2:4]  # Width/height
    targ_conf = targets[..., 4:5]  # Object confidence
    targ_class = targets[..., 5:]  # Class labels
    
    # Convert relative coordinates to absolute coordinates
    anchors = mx.reshape(model.anchors, (1, 1, 1, B, 2))
    
    # Convert predicted boxes to absolute coordinates
    grid_x, grid_y = mx.meshgrid(
        mx.arange(S, dtype=mx.float32),
        mx.arange(S, dtype=mx.float32)
    )
    grid_xy = mx.stack([grid_x, grid_y], axis=-1)
    grid_xy = mx.expand_dims(grid_xy, axis=2)  # Add box dimension
    
    # Convert to absolute coordinates with numerical stability
    pred_xy_abs = (pred_xy + grid_xy) / S
    pred_wh_abs = mx.exp(mx.clip(pred_wh, -10, 10)) * anchors  # Clip exponential
    pred_boxes = mx.concatenate([pred_xy_abs, pred_wh_abs], axis=-1)
    
    # Convert target boxes to absolute coordinates
    targ_xy_abs = (targ_xy + grid_xy) / S
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
    
    # WH loss (using anchor-relative coordinates with better scale handling)
    wh_loss = mx.sum(
        obj_mask * (
            (mx.sqrt(pred_wh_abs + 1e-6) - mx.sqrt(targ_wh_abs + 1e-6)) ** 2
        )
    ) / num_objects
    
    # 2. Object confidence loss
    ious = compute_box_iou(pred_boxes, targ_boxes)
    conf_target = mx.clip(ious * obj_mask, 0.0, 1.0)  # Clip IoU values
    obj_loss = mx.sum(
        obj_mask * (pred_conf - conf_target) ** 2
    ) / num_objects
    
    # 3. No-object confidence loss with focal loss style weighting
    noobj_factor = (1 - pred_conf) ** 2  # Focus more on confident false positives
    noobj_loss = mx.sum(
        noobj_mask * noobj_factor * pred_conf ** 2
    ) / (mx.sum(noobj_mask) + 1e-6)
    
    # 4. Class prediction loss with optional class weights
    if class_weights is not None:
        # Apply class weights to the loss
        class_weights = mx.array(class_weights, dtype=mx.float32)
        class_weights = mx.reshape(class_weights, (1, 1, 1, 1, -1))
        class_loss = mx.sum(
            obj_mask * class_weights * mx.sum(
                mx.clip((pred_class - targ_class) ** 2, 0, 100),  # Clip squared error
                axis=-1,
                keepdims=True
            )
        ) / num_objects
    else:
        class_loss = mx.sum(
            obj_mask * mx.sum(
                mx.clip((pred_class - targ_class) ** 2, 0, 100),  # Clip squared error
                axis=-1,
                keepdims=True
            )
        ) / num_objects
    
    # Scale losses
    coord_loss = lambda_coord * (xy_loss + wh_loss)
    conf_loss = obj_loss + lambda_noobj * noobj_loss
    
    # Combine all losses with their respective weights and clip for stability
    total_loss = mx.clip(
        coord_loss +  # Coordinate loss
        conf_loss +   # Confidence loss
        class_loss,   # Class prediction loss
        -100, 100     # Clip total loss
    )
    
    # Return loss components for logging
    loss_components = {
        'coord': coord_loss.item(),
        'conf': conf_loss.item(),
        'noobj': noobj_loss.item(),
        'class': class_loss.item()
    }
    
    return total_loss, loss_components
