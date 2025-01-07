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
    boxes1, boxes2: [batch_size, num_boxes, 4] where each box is [x, y, w, h]
    Returns: IoU tensor of shape [batch_size, num_boxes]
    """
    # Ensure both inputs have same batch size
    batch_size = boxes1.shape[0]

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
    boxes1_area = boxes1[..., 2:3] * boxes1[..., 3:4]
    boxes2_area = boxes2[..., 2:3] * boxes2[..., 3:4]
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
        raise ValueError(
            f"Predictions must have 4 dimensions, got {len(predictions.shape)}"
        )

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


def yolo_loss(predictions, targets, model):
    """
    Compute YOLO loss with anchor boxes
    predictions: [batch_size, S, S, B*5] - Raw predictions from model
    targets: [batch_size, S, S, 5] - Target boxes
    """
    batch_size = predictions.shape[0]
    S = model.S
    B = model.B

    # Reshape predictions to separate boxes
    pred = mx.reshape(predictions, (batch_size, S, S, B, 5))

    # Extract components
    pred_xy = mx.sigmoid(pred[..., 0:2])  # Center coordinates relative to grid cell
    pred_wh = pred[..., 2:4]  # Width/height predictions (will be scaled by anchors)
    pred_conf = mx.sigmoid(pred[..., 4])  # Object confidence

    # Extract target components
    target_xy = targets[..., 0:2]  # Target center coordinates
    target_wh = targets[..., 2:4]  # Target width/height
    target_conf = targets[..., 4:5]  # Target confidence

    # Create grid cell offsets
    grid_x, grid_y = mx.meshgrid(
        mx.arange(S, dtype=mx.float32), mx.arange(S, dtype=mx.float32)
    )
    grid_xy = mx.stack([grid_x, grid_y], axis=-1)
    grid_xy = mx.expand_dims(grid_xy, axis=2)  # Add box dimension

    # Convert predictions to global coordinates
    pred_xy = (pred_xy + grid_xy) / S  # Add cell offset and normalize

    # Scale width/height by anchors
    anchors = mx.expand_dims(model.anchors, axis=0)  # [1,B,2]
    anchors = mx.expand_dims(anchors, axis=0)  # [1,1,B,2]
    anchors = mx.expand_dims(anchors, axis=0)  # [1,1,1,B,2]
    pred_wh = anchors * mx.exp(pred_wh) / S  # Scale by anchors and normalize

    # Compute IoU between predictions and targets
    pred_boxes = mx.concatenate([pred_xy, pred_wh], axis=-1)
    target_boxes = mx.concatenate([target_xy, target_wh], axis=-1)
    ious = compute_box_iou(pred_boxes, target_boxes)

    # Find best anchor box for each target
    best_ious = mx.max(ious, axis=-1, keepdims=True)
    box_mask = (ious >= best_ious) * mx.expand_dims(target_conf, axis=-1)

    # Compute losses
    xy_loss = box_mask * mx.sum(
        mx.square(pred_xy - mx.expand_dims(target_xy, axis=-2)), axis=-1
    )
    wh_loss = box_mask * mx.sum(
        mx.square(
            mx.log(pred_wh + 1e-6) - mx.expand_dims(mx.log(target_wh + 1e-6), axis=-2)
        ),
        axis=-1,
    )
    conf_loss = box_mask * mx.square(pred_conf - 1) + (1 - box_mask) * mx.square(
        pred_conf
    )

    # Weight the losses
    lambda_coord = 5.0
    lambda_noobj = 0.5
    total_loss = (
        lambda_coord * (xy_loss + wh_loss)
        + conf_loss * (1 - lambda_noobj)
        + lambda_noobj * conf_loss * (1 - box_mask)
    )

    return mx.mean(total_loss), {
        "xy": mx.mean(xy_loss).item(),
        "wh": mx.mean(wh_loss).item(),
        "conf": mx.mean(conf_loss).item(),
        "iou": mx.mean(ious * box_mask).item(),
    }
