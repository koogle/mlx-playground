"""
Verification script for YOLO model predictions.
Loads a single image and its ground truth annotations,
runs the forward pass, and analyzes the predictions in detail.
"""

import os
import argparse
import xml.etree.ElementTree as ET
import mlx.core as mx
from PIL import Image, ImageDraw

from model import YOLO
from data.voc import VOC_CLASSES
from detect import preprocess_image, decode_predictions


def load_ground_truth(annotation_path):
    """Load ground truth annotations from VOC XML file."""
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        objects.append({"class": name, "bbox": [xmin, ymin, xmax, ymax]})

    # Get image size
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    return objects, (width, height)


def analyze_raw_predictions(predictions, S=7, B=2, C=20):
    """Analyze raw predictions from model."""
    print("\nAnalyzing Raw Predictions:")
    print(f"Shape: {predictions.shape}")
    print(f"Value range: [{float(mx.min(predictions)):.3f}, {float(mx.max(predictions)):.3f}]")

    # Reshape to [S, S, B*(5+C)]
    pred = predictions[0]  # [S, S, B*(5+C)]
    print(f"Reshaped predictions shape: {pred.shape}")

    # For each cell
    for i in range(S):
        for j in range(S):
            # Get class probabilities
            class_offset = B * 5
            class_logits = pred[i, j, class_offset:class_offset + C]
            cell_class_probs = mx.softmax(class_logits)  # Convert logits to probabilities

            # For each box
            for b in range(B):
                box_offset = b * 5
                box_end = box_offset + 5
                
                # Get box predictions
                box_values = pred[i, j, box_offset:box_end]
                if len(box_values) < 5:
                    print(f"Warning: Box values truncated at cell ({i},{j}) box {b}")
                    continue
                    
                # Extract components
                tx = mx.sigmoid(box_values[0])  # Center x
                ty = mx.sigmoid(box_values[1])  # Center y
                tw = box_values[2]  # Width
                th = box_values[3]  # Height
                conf = mx.sigmoid(box_values[4])  # Confidence

                # Only print if confidence is significant
                if conf > 0.1:
                    print(f"\nCell ({i},{j}) Box {b}:")
                    print(f"  Raw box values: {[float(v) for v in box_values]}")
                    print(f"  Position: tx={float(tx):.3f}, ty={float(ty):.3f}")
                    print(f"  Size: tw={float(tw):.3f}, th={float(th):.3f}")
                    print(f"  Confidence: {float(conf):.3f}")

                    # Print all class probabilities above threshold
                    print("  Class probabilities:")
                    class_probs_list = [float(p) for p in cell_class_probs]
                    for class_idx, prob in enumerate(class_probs_list):
                        if prob > 0.1:  # Only show significant probabilities
                            print(f"    {VOC_CLASSES[class_idx]}: {prob:.3f}")


def visualize_predictions(image_path, boxes, scores, class_ids, ground_truth=None):
    """Visualize predictions and ground truth on image."""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Draw predictions in red
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        # Convert normalized coordinates to pixel coordinates
        x1 = x1 * img.width
        y1 = y1 * img.height
        x2 = x2 * img.width
        y2 = y2 * img.height

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        label = f"{VOC_CLASSES[int(class_id)]}: {float(score):.2f}"
        draw.text((x1, y1 - 10), label, fill="red")

    # Draw ground truth in green
    if ground_truth:
        for obj in ground_truth:
            x1, y1, x2, y2 = obj["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            draw.text((x1, y1 - 20), obj["class"], fill="green")

    # Save visualization
    output_path = "debug_output.jpg"
    img.save(output_path)
    print(f"\nVisualization saved to {output_path}")


def verify_predictions(predictions, targets, model, iou_threshold=0.5):
    """
    Verify model predictions against ground truth targets.
    Returns IoU scores and detection metrics.
    """
    # Convert predictions from NCHW to NHWC format
    predictions = mx.transpose(predictions, (0, 2, 3, 1))
    
    # Extract dimensions
    batch_size = predictions.shape[0]
    S = model.S
    B = model.B
    C = model.C
    
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
    
    # Compute IoU between predicted and target boxes
    ious = compute_box_iou(
        mx.reshape(pred_boxes, (-1, S*S*B, 4)),
        mx.reshape(targ_boxes, (-1, S*S*B, 4))
    )
    ious = mx.reshape(ious, (batch_size, S, S, B))
    
    # Get predicted and target classes
    pred_classes = mx.argmax(pred_class, axis=-1)
    targ_classes = mx.argmax(targ_class, axis=-1)
    
    # Compute metrics
    obj_mask = targ_conf[..., 0] > 0
    correct_class = (pred_classes == targ_classes) * obj_mask
    correct_box = (ious > iou_threshold) * obj_mask
    correct_obj = (pred_conf[..., 0] > 0.5) * obj_mask
    correct_noobj = (pred_conf[..., 0] <= 0.5) * (1 - obj_mask)
    
    metrics = {
        'mean_iou': mx.mean(ious * obj_mask).item(),
        'class_accuracy': mx.sum(correct_class).item() / (mx.sum(obj_mask).item() + 1e-6),
        'box_accuracy': mx.sum(correct_box).item() / (mx.sum(obj_mask).item() + 1e-6),
        'obj_accuracy': mx.sum(correct_obj).item() / (mx.sum(obj_mask).item() + 1e-6),
        'noobj_accuracy': mx.sum(correct_noobj).item() / (mx.sum(1 - obj_mask).item() + 1e-6)
    }
    
    return ious, metrics


def main():
    parser = argparse.ArgumentParser(description="Verify YOLO model predictions")
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument(
        "--conf-thresh", type=float, default=0.1, help="Confidence threshold"
    )
    parser.add_argument(
        "--class-thresh",
        type=float,
        default=0.1,
        help="Class probability threshold",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model = YOLO()
    weights = mx.load(args.model)
    model.update(weights)

    # Load and preprocess image
    print("Processing image...")
    image, orig_size = preprocess_image(args.image)

    # Get ground truth
    annotation_path = args.image.replace("JPEGImages", "Annotations").replace(
        ".jpg", ".xml"
    )
    if os.path.exists(annotation_path):
        ground_truth, image_size = load_ground_truth(annotation_path)
        print("\nGround Truth:")
        for obj in ground_truth:
            print(f"- {obj['class']} at {obj['bbox']}")
    else:
        print("\nNo ground truth annotations found")
        ground_truth = None

    # Run forward pass
    print("\nRunning forward pass...")
    predictions, features = model(image, return_features=True)

    # Analyze raw predictions
    analyze_raw_predictions(predictions)

    # Decode predictions
    print("\nDecoding predictions...")
    boxes, scores, class_ids = decode_predictions(
        predictions,
        confidence_threshold=args.conf_thresh,
        class_threshold=args.class_thresh,
    )

    # Print decoded predictions
    print("\nDecoded Predictions:")
    for box, score, class_id in zip(boxes, scores, class_ids):
        print(f"- {VOC_CLASSES[int(class_id)]}: {float(score):.3f} at {[float(x) for x in box]}")

    # Verify predictions
    targets = ...  # Load targets
    ious, metrics = verify_predictions(predictions, targets, model)
    print("\nVerification Metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.3f}")

    # Visualize results
    visualize_predictions(args.image, boxes, scores, class_ids, ground_truth)


if __name__ == "__main__":
    main()
