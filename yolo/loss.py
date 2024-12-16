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


def yolo_loss(predictions, targets, lambda_coord=5.0, lambda_noobj=0.5):
    """
    Compute YOLO loss
    Args:
        predictions: Model predictions [batch, S, S, B*5 + C]
        targets: Ground truth [batch, S, S, 5 + C]
        lambda_coord: Weight for coordinate predictions
        lambda_noobj: Weight for no object predictions
    """
    S = predictions.shape[1]  # Grid size
    B = 2  # Number of boxes per cell

    # Reshape predictions to [batch, S, S, B, 5] for boxes and [batch, S, S, C] for classes
    pred_boxes = mx.reshape(predictions[..., : B * 5], (-1, S, S, B, 5))
    pred_classes = predictions[..., B * 5 :]

    # Get target components and expand dimensions for broadcasting
    target_boxes = targets[..., :4]  # [batch, S, S, 4]
    target_boxes = mx.expand_dims(target_boxes, axis=3)  # [batch, S, S, 1, 4]
    target_obj = targets[..., 4]  # [batch, S, S]
    target_obj = mx.expand_dims(target_obj, axis=3)  # [batch, S, S, 1]
    target_classes = targets[..., 5:]  # [batch, S, S, C]

    # Compute IOUs for all predicted boxes with target
    ious = mx.stack(
        [
            compute_iou(pred_boxes[..., i, :4], target_boxes[..., 0, :])
            for i in range(B)
        ],
        axis=-1,
    )  # [batch, S, S, B]

    # Find best box for each target
    best_box_ious = mx.max(ious, axis=-1, keepdims=True)  # [batch, S, S, 1]
    best_box_mask = ious >= best_box_ious  # [batch, S, S, B]

    # Coordinate loss (only for cells with objects)
    coord_mask = mx.repeat(target_obj, B, axis=-1)  # [batch, S, S, B]
    coord_mask = coord_mask * best_box_mask  # [batch, S, S, B]

    # Compute coordinate losses separately
    xy_loss = mx.sum(
        (pred_boxes[..., :2] - target_boxes[..., :2]) ** 2,
        axis=-1,
    )  # [batch, S, S, B]

    wh_loss = mx.sum(
        (
            mx.sqrt(mx.maximum(pred_boxes[..., 2:4], 1e-6))
            - mx.sqrt(mx.maximum(target_boxes[..., 2:4], 1e-6))
        )
        ** 2,
        axis=-1,
    )  # [batch, S, S, B]

    coord_loss = lambda_coord * coord_mask * (xy_loss + wh_loss)

    # Confidence loss
    conf_mask = mx.repeat(target_obj, B, axis=-1)  # [batch, S, S, B]
    noobj_mask = 1 - conf_mask  # [batch, S, S, B]

    conf_loss = (
        conf_mask * (pred_boxes[..., 4] - ious) ** 2  # Object confidence loss
        + lambda_noobj
        * noobj_mask
        * pred_boxes[..., 4] ** 2  # No object confidence loss
    )

    # Class loss (only for cells with objects)
    class_loss = target_obj[..., 0] * mx.sum(
        (pred_classes - target_classes) ** 2, axis=-1
    )

    # Compute total loss
    total_loss = mx.mean(
        mx.sum(coord_loss, axis=-1) + mx.sum(conf_loss, axis=-1) + class_loss
    )

    return total_loss
