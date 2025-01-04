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


def compute_box_iou(pred_boxes, target_boxes):
    """
    Compute IoU between predicted and target boxes.
    Both inputs should have shape [..., 4] where last dim is (x,y,w,h)
    Returns IoU values of same shape as inputs except for last dimension
    """
    # Extract coordinates
    x1 = pred_boxes[..., 0:1]
    y1 = pred_boxes[..., 1:2]
    w1 = pred_boxes[..., 2:3]
    h1 = pred_boxes[..., 3:4]
    
    x2 = target_boxes[..., 0:1]
    y2 = target_boxes[..., 1:2]
    w2 = target_boxes[..., 2:3]
    h2 = target_boxes[..., 3:4]
    
    # Convert to corner coordinates
    x1_min = x1 - w1/2
    y1_min = y1 - h1/2
    x1_max = x1 + w1/2
    y1_max = y1 + h1/2
    
    x2_min = x2 - w2/2
    y2_min = y2 - h2/2
    x2_max = x2 + w2/2
    y2_max = y2 + h2/2
    
    # Calculate intersection area
    inter_x1 = mx.maximum(x1_min, x2_min)
    inter_y1 = mx.maximum(y1_min, y2_min)
    inter_x2 = mx.minimum(x1_max, x2_max)
    inter_y2 = mx.minimum(y1_max, y2_max)
    
    inter_w = mx.maximum(inter_x2 - inter_x1, 0)
    inter_h = mx.maximum(inter_y2 - inter_y1, 0)
    intersection = inter_w * inter_h
    
    # Calculate union area
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-6)
    return mx.squeeze(iou, axis=-1)  # Remove last dimension


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
    pred_class = pred_class - mx.max(pred_class, axis=-1, keepdims=True)
    pred_class = mx.exp(pred_class)
    pred_class = pred_class / (mx.sum(pred_class, axis=-1, keepdims=True) + 1e-10)
    
    # Extract components from targets
    targ_xy = targets[..., 0:2]  # Center coordinates
    targ_wh = targets[..., 2:4]  # Width/height
    targ_conf = targets[..., 4:5]  # Object confidence
    targ_class = targets[..., 5:]  # Class labels
    
    # Convert relative coordinates to absolute coordinates
    anchors = mx.reshape(model.anchors, (1, 1, 1, B, 2))
    
    # Create grid
    grid_x, grid_y = mx.meshgrid(
        mx.arange(S, dtype=mx.float32),
        mx.arange(S, dtype=mx.float32)
    )
    grid_xy = mx.stack([grid_x, grid_y], axis=-1)
    grid_xy = mx.expand_dims(grid_xy, axis=2)  # Add box dimension
    
    # Convert to absolute coordinates
    pred_xy_abs = (pred_xy + grid_xy) / S
    pred_wh_abs = mx.exp(mx.clip(pred_wh, -10, 10)) * anchors
    pred_boxes = mx.concatenate([pred_xy_abs, pred_wh_abs], axis=-1)
    
    targ_xy_abs = (targ_xy + grid_xy) / S
    targ_wh_abs = targ_wh * anchors
    targ_boxes = mx.concatenate([targ_xy_abs, targ_wh_abs], axis=-1)
    
    # Object mask (1 for objects, 0 for no objects)
    obj_mask = targ_conf
    noobj_mask = 1.0 - obj_mask
    
    # Count number of objects for normalization
    num_objects = mx.sum(obj_mask) + 1e-6
    
    # 1. Coordinate loss (only for cells with objects)
    # XY loss with better scale handling
    xy_scale = 2.0 - targ_wh_abs[..., 0:1] * targ_wh_abs[..., 1:2]  # Scale based on box size
    xy_loss = mx.sum(
        obj_mask * xy_scale * ((pred_xy - targ_xy) ** 2)
    ) / num_objects
    
    # WH loss with better scale handling
    wh_scale = 2.0 - targ_wh_abs[..., 0:1] * targ_wh_abs[..., 1:2]  # Scale based on box size
    wh_loss = mx.sum(
        obj_mask * wh_scale * (
            (mx.sqrt(pred_wh_abs + 1e-6) - mx.sqrt(targ_wh_abs + 1e-6)) ** 2
        )
    ) / num_objects
    
    # 2. Object confidence loss with IoU weighting
    ious = compute_box_iou(
        mx.reshape(pred_boxes, (-1, S*S*B, 4)),
        mx.reshape(targ_boxes, (-1, S*S*B, 4))
    )
    ious = mx.reshape(ious, (batch_size, S, S, B))
    ious = mx.expand_dims(ious, axis=-1)
    
    # Use IoU-aware confidence loss
    conf_target = mx.clip(ious, 0.0, 1.0)
    conf_scale = 2.0 - obj_mask  # Focus more on objects
    obj_loss = mx.sum(
        obj_mask * conf_scale * (pred_conf - conf_target) ** 2
    ) / num_objects
    
    # 3. No-object confidence loss with focal loss style weighting
    noobj_factor = (1 - pred_conf) ** 2  # Focus more on confident false positives
    noobj_scale = mx.ones_like(noobj_mask)  # Base scale for no-object predictions
    noobj_loss = mx.sum(
        noobj_mask * noobj_factor * noobj_scale * pred_conf ** 2
    ) / (mx.sum(noobj_mask) + 1e-6)
    
    # 4. Class prediction loss with optional class weights and focal loss
    if class_weights is not None:
        class_weights = mx.array(class_weights, dtype=mx.float32)
        class_weights = mx.reshape(class_weights, (1, 1, 1, 1, -1))
        focal_weight = (1 - pred_class) ** 2  # Focal loss weight
        class_loss = mx.sum(
            obj_mask * class_weights * focal_weight * mx.sum(
                mx.clip((pred_class - targ_class) ** 2, 0, 100),
                axis=-1,
                keepdims=True
            )
        ) / num_objects
    else:
        focal_weight = (1 - pred_class) ** 2  # Focal loss weight
        class_loss = mx.sum(
            obj_mask * focal_weight * mx.sum(
                mx.clip((pred_class - targ_class) ** 2, 0, 100),
                axis=-1,
                keepdims=True
            )
        ) / num_objects
    
    # Scale losses
    coord_loss = lambda_coord * (xy_loss + wh_loss)
    conf_loss = obj_loss + lambda_noobj * noobj_loss
    
    # Combine all losses with their respective weights
    total_loss = mx.clip(
        coord_loss +
        conf_loss +
        class_loss,
        -100, 100
    )
    
    # Return loss components for logging
    loss_components = {
        'coord': coord_loss.item(),
        'conf': conf_loss.item(),
        'noobj': noobj_loss.item(),
        'class': class_loss.item()
    }
    
    return total_loss, loss_components
