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
    
    # Union area - keep dimensions consistent
    boxes1_area = (boxes1[..., 2:3] * boxes1[..., 3:4])
    boxes2_area = (boxes2[..., 2:3] * boxes2[..., 3:4])
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


def yolo_loss(predictions, targets, model, lambda_coord=10.0, lambda_noobj=1.0, class_weights=None):
    """
    Compute YOLO loss
    predictions: [batch_size, B*(5 + C), S, S] - Output from model in NCHW format
    targets: [batch_size, S*S*(5 + C)] - Target in flattened format
    """
    batch_size = predictions.shape[0]
    S = model.S  # Grid size
    B = model.B  # Number of boxes per cell
    C = model.C  # Number of classes
    
    # Reshape predictions from NCHW to NHWC format
    pred = mx.transpose(predictions, (0, 2, 3, 1))  # [batch, S, S, B*(5 + C)]
    
    # Split predictions
    pred_boxes = pred[..., :B*5].reshape(-1, S, S, B, 5)  # [x, y, w, h, conf]
    pred_classes = pred[..., B*5:]  # [class_probs]
    
    # Reshape targets
    target = targets.reshape(-1, S, S, 5 + C)  # [batch, S, S, 5 + C]
    target_boxes = target[..., :5]  # [x, y, w, h, obj]
    target_classes = target[..., 5:]  # [class_probs]
    
    # Compute IoU for each predicted box
    ious = mx.zeros((batch_size, S, S, B))
    target_boxes_expanded = mx.expand_dims(target_boxes[..., :4], axis=3)
    target_boxes_expanded = mx.repeat(target_boxes_expanded, B, axis=3)
    
    # Compute IoU between predictions and targets
    ious = compute_box_iou(
        pred_boxes[..., :4].reshape(-1, S*S*B, 4),
        target_boxes_expanded.reshape(-1, S*S*B, 4)
    ).reshape(-1, S, S, B)
    
    # Find responsible box (box with highest IoU)
    best_box_mask = mx.zeros((batch_size, S, S, B))
    best_ious = mx.argmax(ious, axis=3)
    best_box_mask = mx.scatter_add(best_box_mask, best_ious, mx.ones_like(best_ious))
    
    # Object mask from target
    obj_mask = target_boxes[..., 4:5]
    noobj_mask = 1 - obj_mask
    
    # Box coordinate loss (only for responsible boxes)
    box_loss = obj_mask * mx.sum(
        mx.square(pred_boxes[..., :2] - target_boxes[..., :2]) +  # xy loss
        mx.square(mx.sqrt(pred_boxes[..., 2:4]) - mx.sqrt(target_boxes[..., 2:4])),  # wh loss
        axis=-1
    )
    box_loss = lambda_coord * mx.sum(box_loss)
    
    # Confidence loss
    conf_loss_obj = obj_mask * mx.square(pred_boxes[..., 4] - ious)
    conf_loss_noobj = noobj_mask * mx.square(pred_boxes[..., 4])
    conf_loss = mx.sum(conf_loss_obj + lambda_noobj * conf_loss_noobj)
    
    # Class loss (only for cells with objects)
    # Apply focal loss for class predictions
    alpha = 0.25
    gamma = 2.0
    class_probs = mx.softmax(pred_classes, axis=-1)
    focal_weight = mx.power(1 - class_probs, gamma)
    class_loss = obj_mask * focal_weight * mx.sum(
        -target_classes * mx.log(class_probs + 1e-6),
        axis=-1
    )
    if class_weights is not None:
        class_weights = mx.array(class_weights)
        class_loss = class_loss * mx.sum(target_classes * class_weights, axis=-1, keepdims=True)
    class_loss = mx.sum(class_loss)
    
    # Total loss
    total_loss = (box_loss + conf_loss + class_loss) / batch_size
    
    # Return loss components for logging
    components = {
        'iou': mx.mean(ious * obj_mask).item(),
        'coord': box_loss.item() / batch_size,
        'conf': conf_loss.item() / batch_size,
        'class': class_loss.item() / batch_size,
        'noobj': mx.sum(conf_loss_noobj).item() / batch_size
    }
    
    return total_loss, components
