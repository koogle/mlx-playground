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
    boxes1: [batch,S,S,B,4] - Predicted boxes (only x,y,w,h)
    boxes2: [batch,S,S,1,4] - Target boxes (only x,y,w,h)
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
    """YOLO loss function

    Args:
        predictions: [batch_size, S, S, B*(5+C)] - Raw network outputs
        targets: [batch_size, S, S, 5+C] - Target values
            - [0,1]: x,y relative to cell [0,1]
            - [2,3]: w,h relative to image [0,1]
            - [4]: confidence {0,1}
            - [5:]: class probabilities
        model: YOLO model instance
    """
    batch_size = predictions.shape[0]
    S = model.S
    B = model.B
    C = model.C
    eps = 1e-6

    # 1. Reshape predictions
    pred = mx.reshape(predictions, (batch_size, S, S, B, 5 + C))  # [batch,S,S,B,5+C]

    # 2. Box predictions with sigmoid for x,y and anchors for w,h
    pred_xy = mx.sigmoid(mx.clip(pred[..., 0:2], -10, 10))  # [batch,S,S,B,2]

    # Handle anchor boxes and width/height prediction
    anchor_wh = mx.expand_dims(model.anchors, axis=0)  # [1,B,2]
    anchor_wh = mx.expand_dims(anchor_wh, axis=0)  # [1,1,B,2]
    anchor_wh = mx.expand_dims(anchor_wh, axis=0)  # [1,1,1,B,2]
    anchor_wh = mx.broadcast_to(anchor_wh, (batch_size, S, S, B, 2))

    # Predict width/height with stricter clipping
    pred_wh = anchor_wh * mx.exp(
        mx.clip(pred[..., 2:4], -1.0, 1.0)  # Even more limited range
    )

    # Clip predicted width/height to prevent extremes
    pred_wh = mx.clip(pred_wh, eps, 1.0 - eps)

    pred_conf = mx.sigmoid(mx.clip(pred[..., 4], -5, 5))
    pred_classes = mx.softmax(mx.clip(pred[..., 5:], -10, 10), axis=-1)

    # 3. Target processing
    target_xy = mx.expand_dims(targets[..., 0:2], axis=3)  # [batch,S,S,1,2]
    target_wh = mx.expand_dims(targets[..., 2:4], axis=3)  # [batch,S,S,1,2]
    obj_mask = targets[..., 4]  # [batch,S,S]
    target_classes = mx.expand_dims(targets[..., 5:], axis=3)  # [batch,S,S,1,C]

    # 4. Compute IoU between predictions and targets
    pred_boxes = mx.concatenate(
        [pred_xy, pred_wh], axis=-1
    )  # [batch,S,S,B,4] (only x,y,w,h)

    # Don't broadcast target_wh yet - keep as [batch,S,S,1,2]
    target_boxes = mx.concatenate(
        [target_xy, target_wh], axis=-1
    )  # [batch,S,S,1,4] (only x,y,w,h)

    # Compute IoU between predictions and targets
    ious = compute_box_iou(pred_boxes, target_boxes)  # [batch,S,S,B, 1]
    ious = mx.squeeze(ious, axis=-1)  # [batch,S,S,B]

    # 5. Find responsible predictor (fix dimensions)
    best_ious = mx.max(ious, axis=3, keepdims=True)  # [batch,S,S,1]
    box_mask = ious >= best_ious  # [batch,S,S,B]
    box_mask = mx.expand_dims(box_mask, axis=-1)  # [batch,S,S,B,1] for broadcasting

    # Expand obj_mask for broadcasting
    obj_mask = mx.expand_dims(obj_mask, axis=3)  # [batch,S,S,1]
    obj_mask = mx.broadcast_to(obj_mask, (batch_size, S, S, B))  # [batch,S,S,B]

    # Combine masks - no need to squeeze
    box_mask = box_mask * obj_mask  # [batch,S,S,B]

    # 6. Compute losses with proper broadcasting
    # Coordinate loss (already normalized 0-1)
    xy_loss = box_mask * mx.sum(mx.square(pred_xy - target_xy), axis=-1)

    # Width/height loss with better normalization
    wh_scale_error = mx.log(pred_wh / (target_wh + eps) + eps)
    normalized_wh_error = mx.sigmoid(wh_scale_error)
    centered_error = normalized_wh_error - 0.5
    wh_loss = box_mask * mx.sum(mx.square(centered_error), axis=-1)

    # Confidence loss with better weighting for positive examples
    conf_loss = (
        box_mask * mx.square(pred_conf - ious) * 2.0  # Increased weight for objects
        + (1 - box_mask) * mx.square(pred_conf) * 0.1  # Keep background suppressed
    )

    # Class loss (use original box_mask)
    class_loss = box_mask * mx.sum(mx.square(pred_classes - target_classes), axis=-1)

    # 7. Compute final loss with rebalanced weights
    xy_weight = 5.0
    wh_weight = 5.0
    conf_weight = 5.0  # Increased from 2.0
    class_weight = 1.0

    total_loss = (
        xy_weight * xy_loss
        + wh_weight * wh_loss
        + conf_weight * conf_loss
        + class_weight * class_loss
    )

    # Add more detailed debug info
    debug_info = {
        "wh_scale_error": (wh_scale_error.min().item(), wh_scale_error.max().item()),
        "normalized_wh": (
            normalized_wh_error.min().item(),
            normalized_wh_error.max().item(),
        ),
        "centered_error": (centered_error.min().item(), centered_error.max().item()),
        "pred_wh_range": (pred_wh.min().item(), pred_wh.max().item()),
        "target_wh_range": (target_wh.min().item(), target_wh.max().item()),
        "iou_stats": (ious.min().item(), ious.mean().item(), ious.max().item()),
    }

    # Clip total loss to prevent explosion
    total_loss = mx.clip(total_loss, 0, 1000)

    # Return mean loss and components
    return mx.mean(total_loss), {
        "xy": mx.mean(xy_loss).item(),
        "wh": mx.mean(wh_loss).item(),
        "conf": mx.mean(conf_loss).item(),
        "class": mx.mean(class_loss).item(),
        "iou": mx.mean(ious * box_mask).item(),
    }
