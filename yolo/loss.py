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


def compute_box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    boxes1, boxes2: [batch, num_boxes, 4] where each box is [x, y, w, h]
    """
    # Convert from center format to corner format
    boxes1_x1 = boxes1[..., 0:1] - boxes1[..., 2:3] / 2
    boxes1_y1 = boxes1[..., 1:2] - boxes1[..., 3:4] / 2
    boxes1_x2 = boxes1[..., 0:1] + boxes1[..., 2:3] / 2
    boxes1_y2 = boxes1[..., 1:2] + boxes1[..., 3:4] / 2
    
    boxes2_x1 = boxes2[..., 0:1] - boxes2[..., 2:3] / 2
    boxes2_y1 = boxes2[..., 1:2] - boxes2[..., 3:4] / 2
    boxes2_x2 = boxes2[..., 0:1] + boxes2[..., 2:3] / 2
    boxes2_y2 = boxes2[..., 1:2] + boxes2[..., 3:4] / 2
    
    # Intersection coordinates
    inter_x1 = mx.maximum(boxes1_x1, boxes2_x1)
    inter_y1 = mx.maximum(boxes1_y1, boxes2_y1)
    inter_x2 = mx.minimum(boxes1_x2, boxes2_x2)
    inter_y2 = mx.minimum(boxes1_y2, boxes2_y2)
    
    # Intersection area
    inter_w = mx.maximum(0, inter_x2 - inter_x1)
    inter_h = mx.maximum(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    
    # Union area
    boxes1_area = (boxes1[..., 2] * boxes1[..., 3])
    boxes2_area = (boxes2[..., 2] * boxes2[..., 3])
    union = boxes1_area + boxes2_area - intersection
    
    # IoU
    iou = intersection / (union + 1e-6)
    return mx.clip(iou, 0, 1)


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
    predictions = mx.transpose(predictions, (0, 2, 3, 1))
    
    # Reshape predictions and targets
    predictions = mx.reshape(predictions, (batch_size, S, S, B, 5 + C))
    targets = mx.reshape(targets, (batch_size, S, S, B, 5 + C))
    
    # Extract components from predictions with numerical stability
    pred_xy = mx.sigmoid(mx.clip(predictions[..., 0:2], -10, 10))
    pred_wh = mx.clip(predictions[..., 2:4], -10, 10)
    pred_conf = mx.sigmoid(mx.clip(predictions[..., 4:5], -10, 10))
    pred_class = predictions[..., 5:]
    
    # Apply softmax to class predictions with numerical stability
    pred_class = pred_class - mx.max(pred_class, axis=-1, keepdims=True)
    pred_class = mx.exp(pred_class)
    pred_class = pred_class / (mx.sum(pred_class, axis=-1, keepdims=True) + 1e-10)
    
    # Extract components from targets
    targ_xy = targets[..., 0:2]
    targ_wh = targets[..., 2:4]
    targ_conf = targets[..., 4:5]
    targ_class = targets[..., 5:]
    
    # Convert to absolute coordinates
    grid_x, grid_y = mx.meshgrid(
        mx.arange(S, dtype=mx.float32),
        mx.arange(S, dtype=mx.float32)
    )
    grid_xy = mx.stack([grid_x, grid_y], axis=-1)
    grid_xy = mx.expand_dims(grid_xy, axis=2)
    
    # Convert predictions to absolute coordinates
    pred_xy_abs = (pred_xy + grid_xy) / S
    pred_wh_abs = mx.exp(mx.clip(pred_wh, -10, 10)) * model.anchors
    pred_boxes = mx.concatenate([pred_xy_abs, pred_wh_abs], axis=-1)
    
    # Convert targets to absolute coordinates
    targ_xy_abs = (targ_xy + grid_xy) / S
    targ_wh_abs = targ_wh * model.anchors
    targ_boxes = mx.concatenate([targ_xy_abs, targ_wh_abs], axis=-1)
    
    # Object mask (1 for objects, 0 for no objects)
    obj_mask = targ_conf
    noobj_mask = 1.0 - obj_mask
    
    # Count number of objects for normalization
    num_objects = mx.sum(obj_mask) + 1e-6
    
    # 1. Coordinate loss with IoU weighting
    ious = compute_box_iou(
        mx.reshape(pred_boxes, (-1, S*S*B, 4)),
        mx.reshape(targ_boxes, (-1, S*S*B, 4))
    )
    ious = mx.reshape(ious, (batch_size, S, S, B, 1))
    
    # Scale coordinate loss based on IoU
    coord_scale = (2.0 - ious) * obj_mask
    coord_loss = mx.sum(
        coord_scale * (
            (pred_xy - targ_xy) ** 2 +
            (mx.sqrt(pred_wh_abs + 1e-6) - mx.sqrt(targ_wh_abs + 1e-6)) ** 2
        )
    ) / num_objects
    
    # 2. Object confidence loss with IoU weighting
    conf_target = mx.clip(ious, 0.0, 1.0)
    obj_loss = mx.sum(
        obj_mask * (pred_conf - conf_target) ** 2
    ) / num_objects
    
    # 3. No-object confidence loss with focal loss style weighting
    noobj_factor = (1 - pred_conf) ** 2  # Focus more on confident false positives
    noobj_loss = mx.sum(
        noobj_mask * noobj_factor * pred_conf ** 2
    ) / (mx.sum(noobj_mask) + 1e-6)
    
    # 4. Class prediction loss with focal loss and optional class weights
    if class_weights is not None:
        class_weights = mx.array(class_weights, dtype=mx.float32)
        class_weights = mx.reshape(class_weights, (1, 1, 1, 1, -1))
        focal_weight = (1 - pred_class) ** 2
        class_loss = mx.sum(
            obj_mask * class_weights * focal_weight * mx.sum(
                (pred_class - targ_class) ** 2,
                axis=-1,
                keepdims=True
            )
        ) / num_objects
    else:
        focal_weight = (1 - pred_class) ** 2
        class_loss = mx.sum(
            obj_mask * focal_weight * mx.sum(
                (pred_class - targ_class) ** 2,
                axis=-1,
                keepdims=True
            )
        ) / num_objects
    
    # Combine all losses with better normalization
    total_loss = mx.clip(
        lambda_coord * coord_loss +
        obj_loss +
        lambda_noobj * noobj_loss +
        class_loss,
        -1, 1  # Clip total loss for stability
    )
    
    # Return loss components for logging
    loss_components = {
        'iou': mx.mean(ious * obj_mask).item(),
        'coord': coord_loss.item(),
        'conf': obj_loss.item(),
        'noobj': noobj_loss.item(),
        'class': class_loss.item()
    }
    
    return total_loss, loss_components
