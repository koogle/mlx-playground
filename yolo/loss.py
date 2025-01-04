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
    predictions: [batch_size, S, S, B*5 + C] - Output from model in NHWC format
    targets: [batch_size, S, S, 5 + C] - Target in NHWC format
    """
    batch_size = predictions.shape[0]
    S = model.S  # Grid size
    B = model.B  # Number of boxes per cell
    C = model.C  # Number of classes
    
    print(f"Shapes - predictions: {predictions.shape}, targets: {targets.shape}")
    print(f"Model params - S: {S}, B: {B}, C: {C}")
    
    # Calculate expected sizes
    box_features = B * 5  # Each box has 5 values (x, y, w, h, conf)
    total_features = box_features + C
    print(f"Expected features: boxes={box_features}, total={total_features}")
    
    # Split predictions - predictions are already in NHWC format
    pred_boxes = predictions[..., :box_features].reshape(batch_size, S, S, B, 5)  # [batch, S, S, B, 5]
    pred_classes = predictions[..., box_features:]  # [batch, S, S, C]
    
    # Reshape targets - targets are already in NHWC format
    target = targets  # [batch, S, S, 5 + C]
    target_boxes = target[..., :5]  # [batch, S, S, 5]
    target_classes = target[..., 5:]  # [batch, S, S, C]
    
    # Expand target boxes to match prediction shape
    target_boxes = mx.expand_dims(target_boxes, axis=3)  # [batch, S, S, 1, 5]
    target_boxes = mx.repeat(target_boxes, B, axis=3)    # [batch, S, S, B, 5]
    
    # Compute IoU for each predicted box
    # Reshape boxes for IoU computation
    pred_boxes_reshaped = pred_boxes[..., :4].reshape(batch_size, S*S*B, 4)
    target_boxes_reshaped = target_boxes[..., :4].reshape(batch_size, S*S*B, 4)
    
    # Compute IoU between predictions and targets
    ious = compute_box_iou(pred_boxes_reshaped, target_boxes_reshaped)  # [batch, S*S*B]
    ious = ious.reshape(batch_size, S, S, B)  # [batch, S, S, B]
    
    # Find responsible box (box with highest IoU)
    # Instead of using array indexing, we'll use one-hot encoding with max
    best_ious_idx = mx.argmax(ious, axis=3)  # [batch, S, S]
    best_box_mask = mx.zeros((batch_size, S, S, B))
    
    # Create one-hot encoding for best box using comparison
    for b in range(B):
        best_box_mask = mx.where(
            mx.expand_dims(best_ious_idx == b, axis=-1),
            mx.ones_like(best_box_mask),
            best_box_mask
        )
    
    # Object mask from target
    obj_mask = mx.expand_dims(target_boxes[..., 4], axis=-1)  # [batch, S, S, B, 1]
    noobj_mask = 1 - obj_mask
    
    # Box coordinate loss (only for responsible boxes)
    box_loss = obj_mask * mx.sum(
        mx.square(pred_boxes[..., :2] - target_boxes[..., :2]) +  # xy loss
        mx.square(mx.sqrt(pred_boxes[..., 2:4] + 1e-6) - mx.sqrt(target_boxes[..., 2:4] + 1e-6)),  # wh loss
        axis=-1
    )
    box_loss = lambda_coord * mx.sum(box_loss)
    
    # Confidence loss
    conf_loss_obj = obj_mask * mx.square(pred_boxes[..., 4:5] - mx.expand_dims(ious, axis=-1))
    conf_loss_noobj = noobj_mask * mx.square(pred_boxes[..., 4:5])
    conf_loss = mx.sum(conf_loss_obj + lambda_noobj * conf_loss_noobj)
    
    # Class loss (only for cells with objects)
    # Apply focal loss for class predictions
    alpha = 0.25
    gamma = 2.0
    class_probs = mx.softmax(pred_classes, axis=-1)
    focal_weight = mx.power(1 - class_probs + 1e-6, gamma)
    
    # Reshape obj_mask for class loss
    obj_mask_class = mx.squeeze(target_boxes[..., 4:5], axis=-1)  # [batch, S, S, B]
    obj_mask_class = mx.max(obj_mask_class, axis=-1, keepdims=True)  # [batch, S, S, 1]
    
    class_loss = obj_mask_class * focal_weight * mx.sum(
        -target_classes * mx.log(class_probs + 1e-6),
        axis=-1,
        keepdims=True
    )
    if class_weights is not None:
        class_weights = mx.array(class_weights)
        class_loss = class_loss * mx.sum(target_classes * class_weights, axis=-1, keepdims=True)
    class_loss = mx.sum(class_loss)
    
    # Total loss
    total_loss = (box_loss + conf_loss + class_loss) / batch_size
    
    # Return loss components for logging
    components = {
        'iou': mx.mean(ious * mx.squeeze(obj_mask, axis=-1)).item(),
        'coord': box_loss.item() / batch_size,
        'conf': conf_loss.item() / batch_size,
        'class': class_loss.item() / batch_size,
        'noobj': mx.sum(conf_loss_noobj).item() / batch_size
    }
    
    return total_loss, components
