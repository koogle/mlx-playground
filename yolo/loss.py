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
    pred_xy = mx.sigmoid(mx.clip(pred[..., 0:2], -10, 10))  # Cell-relative [0,1]
    pred_wh = model.anchors * mx.exp(
        mx.clip(pred[..., 2:4], -10, 10)
    )  # Image-relative [0,1]
    pred_conf = mx.sigmoid(mx.clip(pred[..., 4], -10, 10))  # Object confidence
    pred_classes = mx.softmax(
        mx.clip(pred[..., 5:], -10, 10), axis=-1
    )  # Class probabilities

    # 3. Target processing
    target_xy = mx.expand_dims(targets[..., 0:2], axis=3)  # [batch,S,S,1,2]
    target_wh = mx.expand_dims(targets[..., 2:4], axis=3)  # [batch,S,S,1,2]
    obj_mask = targets[..., 4]  # [batch,S,S]
    target_classes = mx.expand_dims(targets[..., 5:], axis=3)  # [batch,S,S,1,C]

    # 4. Compute IoU between predictions and targets
    pred_boxes = mx.concatenate([pred_xy, pred_wh], axis=-1)  # [batch,S,S,B,4]
    target_boxes = mx.concatenate([target_xy, target_wh], axis=-1)  # [batch,S,S,1,4]
    ious = compute_box_iou(pred_boxes, target_boxes)  # [batch,S,S,B]

    # 5. Find responsible predictor
    best_ious = mx.max(ious, axis=3, keepdims=True)  # [batch,S,S,1]
    obj_mask = mx.expand_dims(obj_mask, axis=-1)  # [batch,S,S,1]
    box_mask = mx.expand_dims((ious >= best_ious), axis=-1) * mx.expand_dims(
        obj_mask, axis=3
    )  # [batch,S,S,B,1]

    # 6. Compute losses
    # Coordinate loss (only for responsible predictors)
    xy_loss = box_mask * mx.sum(mx.square(pred_xy - target_xy), axis=-1)
    wh_loss = box_mask * mx.sum(mx.square(pred_wh - target_wh), axis=-1)

    # Confidence loss
    conf_loss = (
        box_mask * mx.square(pred_conf - 1)  # Object confidence loss
        + (1 - box_mask) * mx.square(pred_conf) * 0.5  # No object confidence loss
    )

    # Class loss (only for cells with objects)
    class_loss = obj_mask * mx.sum(mx.square(pred_classes - target_classes), axis=-1)

    # 7. Compute final loss with weights
    total_loss = (
        5.0 * (xy_loss + wh_loss)  # Coordinate loss weight
        + conf_loss  # Confidence loss
        + class_loss  # Class loss
    )

    # Compute mean losses for monitoring
    num_objects = mx.sum(obj_mask) + eps

    return mx.mean(total_loss), {
        "xy": mx.mean(xy_loss).item(),
        "wh": mx.mean(wh_loss).item(),
        "conf": mx.mean(conf_loss).item(),
        "class": mx.mean(class_loss).item(),
        "iou": mx.mean(ious * box_mask).item(),
    }
