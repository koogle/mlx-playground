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
    boxes1: [batch,S,S,B,4] - Predicted boxes
    boxes2: [batch,S,S,B,4] - Target boxes (broadcasted)
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

    # Calculate intersection
    inter_x1 = mx.maximum(boxes1_x1, boxes2_x1)
    inter_y1 = mx.maximum(boxes1_y1, boxes2_y1)
    inter_x2 = mx.minimum(boxes1_x2, boxes2_x2)
    inter_y2 = mx.minimum(boxes1_y2, boxes2_y2)

    inter_w = mx.maximum(inter_x2 - inter_x1, 0)
    inter_h = mx.maximum(inter_y2 - inter_y1, 0)
    intersection = inter_w * inter_h

    # Calculate union
    boxes1_area = boxes1[..., 2:3] * boxes1[..., 3:4]
    boxes2_area = boxes2[..., 2:3] * boxes2[..., 3:4]
    union = boxes1_area + boxes2_area - intersection

    return intersection / (union + 1e-6)


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
    predictions: [batch_size, S, S, B*(5+C)] - Raw predictions from model
    targets: [batch_size, S, S, 5+C] - Target boxes and class labels
    """
    batch_size = predictions.shape[0]
    S = model.S
    B = model.B
    C = model.C

    # 1. Reshape predictions
    pred = mx.reshape(predictions, (batch_size, S, S, B, 5 + C))  # [batch,S,S,B,5+C]
    pred_xy = mx.sigmoid(pred[..., 0:2])  # [batch,S,S,B,2]
    pred_wh = pred[..., 2:4]  # [batch,S,S,B,2]
    pred_conf = mx.sigmoid(pred[..., 4])  # [batch,S,S,B]
    pred_classes = mx.softmax(pred[..., 5:], axis=-1)  # [batch,S,S,B,C]

    # 2. Process targets - expand for comparison
    target_xy = mx.expand_dims(targets[..., 0:2], axis=3)  # [batch,S,S,1,2]
    target_wh = mx.expand_dims(targets[..., 2:4], axis=3)  # [batch,S,S,1,2]
    target_conf = targets[..., 4]  # [batch,S,S]
    target_classes = targets[..., 5:]  # [batch,S,S,C]
    target_classes = mx.expand_dims(target_classes, axis=3)  # [batch,S,S,1,C]

    # 3. Grid cell offsets
    grid_x, grid_y = mx.meshgrid(
        mx.arange(S, dtype=mx.float32), mx.arange(S, dtype=mx.float32)
    )
    grid_xy = mx.stack([grid_x, grid_y], axis=-1)  # [S,S,2]
    grid_xy = mx.expand_dims(grid_xy, axis=2)  # [S,S,1,2]

    # 4. Convert predictions to global coordinates
    pred_xy = (pred_xy + grid_xy) / S

    # Scale width/height by anchors
    anchors = mx.expand_dims(model.anchors, axis=0)  # [1,B,2]
    anchors = mx.expand_dims(anchors, axis=0)  # [1,1,B,2]
    anchors = mx.expand_dims(anchors, axis=0)  # [1,1,1,B,2]
    pred_wh = anchors * mx.exp(pred_wh) / S

    # 5. Compute IoU
    pred_boxes = mx.concatenate([pred_xy, pred_wh], axis=-1)  # [batch,S,S,B,4]
    target_boxes = mx.concatenate([target_xy, target_wh], axis=-1)  # [batch,S,S,1,4]
    ious = compute_box_iou(pred_boxes, target_boxes)  # [batch,S,S,B]

    # 6. Create box mask
    best_ious = mx.max(ious, axis=-1, keepdims=True)  # [batch,S,S,1]
    box_mask = ious >= best_ious  # [batch,S,S,B]

    # Expand target_conf for broadcasting
    target_conf = mx.expand_dims(target_conf, axis=-1)  # [batch,S,S,1]
    target_conf = mx.expand_dims(target_conf, axis=3)  # [batch,S,S,1,1]
    target_conf = mx.broadcast_to(
        target_conf, (batch_size, S, S, B, 1)
    )  # [batch,S,S,B,1]

    # Create final mask
    box_mask = box_mask * target_conf  # [batch,S,S,B,1]

    # 7. Compute losses
    xy_loss = box_mask * mx.sum(mx.square(pred_xy - target_xy), axis=-1, keepdims=True)
    wh_loss = box_mask * mx.sum(
        mx.square(mx.log(pred_wh + 1e-6) - mx.log(target_wh + 1e-6)),
        axis=-1,
        keepdims=True,
    )

    pred_conf = mx.expand_dims(pred_conf, axis=-1)

    # Confidence loss
    conf_loss = box_mask * mx.square(pred_conf - 1) + (1 - box_mask) * mx.square(
        pred_conf
    )

    # Class loss (only for cells with objects)
    # class_mask = mx.expand_dims(target_conf, axis=-1)  # [batch,S,S,1,1]
    class_loss = target_conf * mx.sum(
        mx.square(pred_classes - target_classes), axis=-1, keepdims=True
    )

    # 8. Weight and combine losses
    lambda_coord = 5.0
    lambda_noobj = 0.5
    lambda_class = 1.0
    total_loss = (
        lambda_coord * (xy_loss + wh_loss)
        + conf_loss * (1 - lambda_noobj)
        + lambda_noobj * conf_loss * (1 - box_mask)
        + lambda_class * class_loss
    )

    return mx.mean(total_loss), {
        "xy": mx.mean(xy_loss).item(),
        "wh": mx.mean(wh_loss).item(),
        "conf": mx.mean(conf_loss).item(),
        "class": mx.mean(class_loss).item(),
        "iou": mx.mean(mx.squeeze(ious, axis=-1) * box_mask[..., 0]).item(),
    }
