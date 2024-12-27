import mlx.core as mx


def compute_iou(box1, box2):
    """
    Compute IOU between two boxes
    box format: [x, y, w, h]
    """
    # Get coordinates of intersection rectangle
    x1 = mx.maximum(box1[..., 0] - box1[..., 2] / 2, box2[..., 0] - box2[..., 2] / 2)
    y1 = mx.maximum(box1[..., 1] - box1[..., 3] / 2, box2[..., 1] - box2[..., 3] / 2)
    x2 = mx.minimum(box1[..., 0] + box1[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2)
    y2 = mx.minimum(box1[..., 1] + box1[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2)

    # Compute intersection area
    intersection = mx.maximum(0.0, x2 - x1) * mx.maximum(0.0, y2 - y1)

    # Compute union area
    box1_area = box1[..., 2] * box1[..., 3]
    box2_area = box2[..., 2] * box2[..., 3]
    union = box1_area + box2_area - intersection

    return intersection / (union + 1e-6)


def yolo_loss(predictions, targets, model, lambda_coord=5.0, lambda_noobj=0.5):
    """
    Compute YOLOv2 loss with anchor boxes
    Args:
        predictions: Model predictions [batch, S, S, B*(5 + C)]
        targets: Ground truth [batch, S, S, 5 + C]
        model: YOLO model instance (needed for anchor boxes)
        lambda_coord: Weight for coordinate predictions
        lambda_noobj: Weight for no object predictions
    """
    batch_size = predictions.shape[0]
    S = predictions.shape[1]  # Grid size
    B = model.B  # Number of anchor boxes per cell
    C = model.C  # Number of classes

    # Reshape predictions to [batch, S, S, B, 5 + C]
    pred = mx.reshape(predictions, (-1, S, S, B, 5 + C))

    # Split predictions
    pred_xy = mx.sigmoid(pred[..., 0:2])  # Center coordinates
    pred_wh = mx.exp(pred[..., 2:4])  # Width and height
    pred_conf = mx.sigmoid(pred[..., 4:5])  # Object confidence
    pred_class = pred[..., 5:]  # Class probabilities

    # Get target components
    target_boxes = targets[..., :4]  # [batch, S, S, 4]
    target_obj = targets[..., 4:5]  # [batch, S, S, 1]
    target_class = targets[..., 5:]  # [batch, S, S, C]

    # Expand dimensions for broadcasting with anchor boxes
    target_boxes = mx.expand_dims(target_boxes, axis=3)  # [batch, S, S, 1, 4]
    target_obj = mx.expand_dims(target_obj, axis=3)  # [batch, S, S, 1, 1]

    # Create grid offsets
    grid_x = mx.arange(S, dtype=mx.float32)
    grid_y = mx.arange(S, dtype=mx.float32)
    grid_x, grid_y = mx.meshgrid(grid_x, grid_y)
    grid_x = mx.expand_dims(grid_x, axis=-1)
    grid_y = mx.expand_dims(grid_y, axis=-1)
    
    # Add grid offsets to predictions
    pred_xy_abs = (pred_xy + mx.stack([grid_x, grid_y], axis=-1)) / S
    
    # Scale width and height by anchors
    pred_wh_abs = pred_wh * model.anchors

    # Combine predictions for IOU calculation
    pred_boxes_abs = mx.concatenate([pred_xy_abs, pred_wh_abs], axis=-1)

    # Compute IOUs between predicted boxes and target
    ious = mx.stack(
        [compute_iou(pred_boxes_abs[..., i, :], target_boxes[..., 0, :])
         for i in range(B)],
        axis=-1
    )  # [batch, S, S, B]

    # Find best anchor box for each target
    best_box_ious = mx.max(ious, axis=-1, keepdims=True)  # [batch, S, S, 1]
    best_box_mask = (ious >= best_box_ious) * target_obj  # [batch, S, S, B]
    best_box_mask = mx.expand_dims(best_box_mask, axis=-1)  # [batch, S, S, B, 1]

    # Coordinate loss (only for responsible boxes)
    xy_loss = mx.sum(
        best_box_mask * (pred_xy - (target_boxes[..., :2] * S - mx.stack([grid_x, grid_y], axis=-1))) ** 2,
        axis=(-1, -2)
    )

    wh_loss = mx.sum(
        best_box_mask * (
            mx.log(pred_wh + 1e-6) - 
            mx.log(target_boxes[..., 2:4] / model.anchors + 1e-6)
        ) ** 2,
        axis=(-1, -2)
    )

    # Confidence loss
    conf_mask = best_box_mask[..., 0]  # [batch, S, S, B]
    noobj_mask = 1 - conf_mask  # [batch, S, S, B]

    conf_loss = mx.sum(
        conf_mask * (pred_conf[..., 0] - ious) ** 2 +  # Object confidence loss
        lambda_noobj * noobj_mask * pred_conf[..., 0] ** 2,  # No object confidence loss
        axis=(-1)
    )

    # Class loss (only for cells with objects)
    class_loss = mx.sum(
        target_obj[..., 0] * mx.sum(
            mx.softmax(pred_class, axis=-1) - target_class
        ) ** 2,
        axis=(-1)
    )

    # Compute total loss
    total_loss = mx.mean(
        lambda_coord * (xy_loss + wh_loss) + conf_loss + class_loss
    )

    return total_loss
