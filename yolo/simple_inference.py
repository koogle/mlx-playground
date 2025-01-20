import os
import cv2
import time
import mlx.core as mx
import numpy as np
from model import YOLO
import argparse
from pathlib import Path
from PIL import Image, ImageDraw

from data.voc import VOC_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Inference")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model checkpoint (.safetensors file)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Inference interval in seconds",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Image ID or path for inference (e.g., '2008_000008' or full path)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="./VOCdevkit/VOC2012",
        help="Path to VOC dataset directory",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.2,
        help="Confidence threshold for filtering predictions",
    )
    return parser.parse_args()


def preprocess_frame(frame, target_size=448):
    """Preprocess frame for model input"""

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize frame
    resized = cv2.resize(rgb_frame, (target_size, target_size))

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # Add batch dimension
    batched = np.expand_dims(normalized, axis=0)

    return mx.array(batched)


def decode_predictions(predictions, model):
    """Decode raw predictions to bounding boxes and class predictions."""
    print("\nDecoding predictions:")
    print(f"Raw predictions shape: {predictions.shape}")

    S = model.S  # Grid size (7)
    B = model.B  # Number of boxes per cell (5)
    C = model.C  # Number of classes (20)

    # Reshape predictions to [S, S, B, 5+C]
    pred = mx.reshape(predictions, (S, S, B, 5 + C))
    mx.eval(pred)

    # Extract components
    pred_xy = mx.sigmoid(pred[..., :2])  # Sigmoid for relative cell position [0,1]
    pred_wh = pred[..., 2:4]  # Raw width/height
    pred_conf = mx.sigmoid(pred[..., 4])  # Confidence score
    pred_classes = mx.softmax(pred[..., 5:], axis=-1)  # Class probabilities
    mx.eval(pred_xy, pred_wh, pred_conf, pred_classes)

    print("\nAfter sigmoid/softmax:")
    print(f"XY range: [{pred_xy.min().item():.3f}, {pred_xy.max().item():.3f}]")
    print(f"WH range: [{pred_wh.min().item():.3f}, {pred_wh.max().item():.3f}]")
    print(
        f"Confidence range: [{pred_conf.min().item():.3f}, {pred_conf.max().item():.3f}]"
    )
    print(
        f"Class scores range: [{pred_classes.min().item():.3f}, {pred_classes.max().item():.3f}]"
    )

    # Create grid
    grid_x, grid_y = mx.meshgrid(
        mx.arange(S, dtype=mx.float32), mx.arange(S, dtype=mx.float32)
    )
    grid_xy = mx.stack([grid_x, grid_y], axis=-1)
    grid_xy = mx.expand_dims(grid_xy, axis=2)  # Add box dimension
    mx.eval(grid_xy)

    # Convert predictions to absolute coordinates
    cell_size = 1.0 / S
    pred_xy = (pred_xy + grid_xy) * cell_size  # Add cell offset and scale
    pred_wh = mx.sigmoid(pred_wh)  # Sigmoid for relative image size [0,1]
    mx.eval(pred_xy, pred_wh)

    # Convert to corner format (x1, y1, x2, y2)
    pred_x1y1 = pred_xy - pred_wh / 2
    pred_x2y2 = pred_xy + pred_wh / 2
    boxes = mx.concatenate([pred_x1y1, pred_x2y2], axis=-1)

    # Clip to [0,1]
    boxes = mx.clip(boxes, 0.0, 1.0)

    # Get class predictions
    class_scores = mx.max(pred_classes, axis=-1)  # Best class probability
    class_ids = mx.argmax(pred_classes, axis=-1)  # Class with highest probability

    # Combine confidence with class probability
    scores = pred_conf * class_scores

    # Reshape outputs - using tuples for shapes
    boxes = mx.reshape(boxes, (S * S * B, 4))
    scores = mx.reshape(scores, (S * S * B,))  # Fixed: Use tuple with single element
    class_ids = mx.reshape(
        class_ids, (S * S * B,)
    )  # Fixed: Use tuple with single element
    mx.eval(boxes, scores, class_ids)

    print("\nFinal shapes:")
    print(f"Boxes: {boxes.shape}")
    print(f"Scores: {scores.shape}")
    print(f"Class IDs: {class_ids.shape}")

    return boxes, scores, class_ids


def draw_boxes(
    image, boxes, scores, classes=None, class_names=None, color="red", is_gt=False
):
    """Draw boxes on image with class labels

    Args:
        image: PIL Image to draw on
        boxes: List of [x1,y1,x2,y2] boxes in normalized coordinates
        scores: List of confidence scores (or None for ground truth)
        classes: List of class indices
        class_names: List of class names
        color: Color to use for boxes ("red" for predictions, "green" for ground truth)
        is_gt: Whether these are ground truth boxes
    """
    draw = ImageDraw.Draw(image)

    for i, (box, score) in enumerate(
        zip(boxes, scores if scores is not None else [1.0] * len(boxes))
    ):
        # Convert normalized coordinates to pixels
        x1, y1, x2, y2 = box.tolist()
        x1 = x1 * image.width
        y1 = y1 * image.height
        x2 = x2 * image.width
        y2 = y2 * image.height

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label
        if is_gt:
            label = (
                f"GT: {class_names[classes[i].item()]}" if classes is not None else "GT"
            )
        else:
            label = (
                f"{class_names[classes[i].item()]}: {score:.2f}"
                if classes is not None
                else f"{score:.2f}"
            )

        draw.text((x1, y1 - 10), label, fill=color)

    return image


def visualize_activations(frame, predictions, model):
    """Create semi-transparent activation overlay on the frame"""
    batch_size, channels, grid_size, _ = predictions.shape
    height, width = frame.shape[:2]

    # Reshape predictions to [batch, S, S, B*(5+C)]
    pred = mx.reshape(predictions, (batch_size, model.S, model.S, -1))[0]
    mx.eval(pred)

    # Get confidence scores from last layer
    conf_scores = mx.sigmoid(pred[..., 4::5])  # Get all confidence scores
    activation_map = mx.max(conf_scores, axis=-1)  # Max confidence at each cell

    # Convert to numpy and normalize
    act_np = activation_map.tolist()
    act_np = np.array(act_np)

    # Normalize to [0, 1]
    act_min = np.min(act_np)
    act_max = np.max(act_np)
    if act_max > act_min:
        act_np = (act_np - act_min) / (act_max - act_min)

    # Resize to frame size
    act_resized = cv2.resize(act_np, (width, height), interpolation=cv2.INTER_LINEAR)

    # Create heatmap
    heatmap = cv2.applyColorMap((act_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Create overlay
    alpha = 0.5  # Transparency factor
    overlay = cv2.addWeighted(frame, 1, heatmap, alpha, 0)

    # Add text for max activation value
    cv2.putText(
        overlay,
        f"Max activation: {act_max:.3f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    return overlay


def load_ground_truth(image_id, data_dir="./VOCdevkit/VOC2012"):
    """Load ground truth boxes for an image"""
    from xml.etree import ElementTree as ET

    # Load annotation
    anno_path = os.path.join(data_dir, "Annotations", f"{image_id}.xml")
    tree = ET.parse(anno_path)
    root = tree.getroot()

    # Get image size
    size = root.find("size")
    width = float(size.find("width").text)
    height = float(size.find("height").text)

    # Extract boxes
    boxes = []
    classes = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        # Convert to normalized coordinates
        x1 = float(bbox.find("xmin").text) / width
        y1 = float(bbox.find("ymin").text) / height
        x2 = float(bbox.find("xmax").text) / width
        y2 = float(bbox.find("ymax").text) / height
        boxes.append([x1, y1, x2, y2])
        classes.append(obj.find("name").text)

    return np.array(boxes), classes


def analyze_single_image(args, model, image_id):
    """Process a single image with the model and show detailed analysis"""
    # Load and preprocess image
    image_path = args.data_dir / "JPEGImages" / f"{image_id}.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image with PIL for better visualization
    pil_image = Image.open(image_path)

    # Convert to numpy for preprocessing
    np_image = np.array(pil_image)
    input_tensor = preprocess_frame(np_image)

    # Run inference with debug info
    predictions = model(input_tensor)

    print("\nPrediction Analysis:")
    print(f"Raw prediction shape: {predictions.shape}")

    # Analyze raw predictions
    pred = mx.reshape(predictions[0], (model.S, model.S, model.B, 5 + model.C))

    print("\nRaw Prediction Stats:")
    print(
        f"XY (pre-sigmoid): [{pred[..., 0:2].min().item():.3f}, {pred[..., 0:2].max().item():.3f}]"
    )
    print(
        f"WH (pre-exp): [{pred[..., 2:4].min().item():.3f}, {pred[..., 2:4].max().item():.3f}]"
    )
    print(
        f"Conf (pre-sigmoid): [{pred[..., 4].min().item():.3f}, {pred[..., 4].max().item():.3f}]"
    )

    # After activation
    pred_xy = mx.sigmoid(pred[..., 0:2])
    pred_wh = model.anchors * mx.exp(pred[..., 2:4])
    pred_conf = mx.sigmoid(pred[..., 4])

    print("\nActivated Prediction Stats:")
    print(
        f"XY (post-sigmoid): [{pred_xy.min().item():.3f}, {pred_xy.max().item():.3f}]"
    )
    print(f"WH (post-exp): [{pred_wh.min().item():.3f}, {pred_wh.max().item():.3f}]")
    print(
        f"Conf (post-sigmoid): [{pred_conf.min().item():.3f}, {pred_conf.max().item():.3f}]"
    )

    # Process predictions
    boxes, scores, classes = process_predictions(predictions[0], model)

    # Load ground truth
    gt_boxes, gt_classes = load_ground_truth(image_id, args.data_dir)
    gt_class_indices = [
        VOC_CLASSES.index(c) for c in gt_classes
    ]  # Convert class names to indices

    print("\nBox Statistics:")
    print("Ground Truth Boxes:")
    for box, cls in zip(gt_boxes, gt_classes):
        w = box[2] - box[0]
        h = box[3] - box[1]
        print(
            f"- {cls}: w={w:.3f}, h={h:.3f}, center=({(box[0]+box[2])/2:.3f}, {(box[1]+box[3])/2:.3f})"
        )

    print("\nPredicted Boxes:")
    for box, score, class_id in zip(boxes, scores, classes):
        w = box[2].item() - box[0].item()
        h = box[3].item() - box[1].item()
        cls = VOC_CLASSES[class_id.item()]
        print(
            f"- {cls} ({score.item():.3f}): w={w:.3f}, h={h:.3f}, center=({(box[0].item()+box[2].item())/2:.3f}, {(box[1].item()+box[3].item())/2:.3f})"
        )

    # Draw ground truth boxes in green
    output_image = draw_boxes(
        pil_image,
        mx.array(gt_boxes),
        None,  # No scores for ground truth
        mx.array(gt_class_indices),  # Now gt_class_indices is defined
        VOC_CLASSES,
        color="green",
        is_gt=True,
    )

    # Draw predicted boxes in red
    output_image = draw_boxes(
        output_image, boxes, scores, classes, VOC_CLASSES, color="red", is_gt=False
    )

    # Save or display result
    output_path = f"output_{image_id}.jpg"
    output_image.save(output_path)
    print(f"\nSaved detection result to {output_path}")

    # Print detection details
    print("\nGround Truth:")
    for box, class_name in zip(gt_boxes, gt_classes):
        print(f"- {class_name} at {box.tolist()}")

    print("\nDetections:")
    for box, score, class_id in zip(boxes, scores, classes):
        class_name = VOC_CLASSES[class_id.item()]
        print(f"- {class_name}: {score.item():.3f} at {box.tolist()}")


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    # Convert to (x1, y1, x2, y2) format if needed
    box1 = np.array(box1)
    box2 = np.array(box2)

    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)


def main():
    args = parse_args()
    print("Loading model...")

    # Initialize model with same parameters as training
    model = YOLO(S=7, B=5, C=20)

    if not args.model.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    try:
        model.load_weights(str(args.model))
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("\nModel architecture:")
        print(f"Grid size (S): {model.S}")
        print(f"Boxes per cell (B): {model.B}")
        print(f"Number of classes (C): {model.C}")
        print(
            f"Output channels: {model.B * (5 + model.C)}"
        )  # 5 box params + C class scores
        return

    model.eval()
    print(f"Loaded model from {args.model}")

    if args.image:
        image_id = args.image.stem
        analyze_single_image(args, model, image_id)
        return

    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    last_inference_time = 0
    inference_interval = args.interval
    show_activations = False

    print("Starting inference. Press 'q' to quit, TAB to toggle activation view")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()

            # Run inference every interval
            if current_time - last_inference_time >= inference_interval:
                try:
                    input_tensor = preprocess_frame(frame)
                    predictions = model(input_tensor)

                    # Process predictions
                    boxes, scores, classes = process_predictions(
                        predictions[0], model, conf_threshold=args.conf_threshold
                    )

                    # Convert frame to PIL Image for drawing
                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # Draw detections
                    output_frame = draw_boxes(
                        pil_frame, boxes, scores, classes, VOC_CLASSES
                    )

                    # Convert back to OpenCV format
                    frame = cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR)

                    # Add activation overlay if enabled
                    if show_activations:
                        frame = visualize_activations(frame, predictions, model)

                    # Show frame
                    cv2.imshow("YOLO Detection", frame)

                    # Print detection details
                    print("\nDetections:")
                    for box, score, class_id in zip(boxes, scores, classes):
                        class_name = VOC_CLASSES[class_id.item()]
                        print(f"- {class_name}: {score.item():.3f} at {box.tolist()}")

                    last_inference_time = current_time

                except Exception as e:
                    print(f"Inference error: {e}")

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("\t"):  # TAB key
                show_activations = not show_activations
                print(
                    f"Activation overlay {'enabled' if show_activations else 'disabled'}"
                )

    finally:
        cap.release()
        cv2.destroyAllWindows()


def process_predictions(predictions, model, conf_threshold=0.15):
    """Process raw predictions into final detections."""
    boxes, scores, classes = decode_predictions(predictions, model)

    # Convert to numpy for filtering
    boxes_np = np.array([b.tolist() for b in boxes])
    scores_np = np.array([s.tolist() for s in scores])
    classes_np = np.array([c.tolist() for c in classes])

    # Print confidence scores for debugging
    print("\nRaw confidence scores:")
    print(f"Min: {scores_np.min():.3f}, Max: {scores_np.max():.3f}")
    print(f"Mean confidence: {scores_np.mean():.3f}")
    print(f"Confidence threshold: {conf_threshold}")

    # Filter by confidence
    conf_mask = scores_np > conf_threshold
    filtered_boxes = boxes_np[conf_mask]
    filtered_scores = scores_np[conf_mask]
    filtered_classes = classes_np[conf_mask]

    print("After confidence filtering:")
    print(f"Number of boxes above threshold: {len(filtered_scores)}")
    if len(filtered_scores) > 0:
        print(
            f"Remaining confidence range: [{filtered_scores.min():.3f}, {filtered_scores.max():.3f}]"
        )

    if len(filtered_scores) == 0:
        return mx.array([]), mx.array([]), mx.array([])

    # Convert back to MLX arrays
    return (
        mx.array(filtered_boxes),
        mx.array(filtered_scores),
        mx.array(filtered_classes),
    )


if __name__ == "__main__":
    main()
