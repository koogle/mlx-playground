import os
import cv2
import time
import mlx.core as mx
import numpy as np
from model import YOLO
import argparse
from pathlib import Path
from PIL import Image, ImageDraw


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
    return parser.parse_args()


def preprocess_frame(frame, target_size=448):
    """Preprocess frame for model input"""
    print("\nPreprocessing:")
    print(f"Input frame shape: {frame.shape}")

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize frame
    resized = cv2.resize(rgb_frame, (target_size, target_size))
    print(f"Resized shape: {resized.shape}")

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    print(f"Value range: [{normalized.min():.3f}, {normalized.max():.3f}]")

    # Add batch dimension
    batched = np.expand_dims(normalized, axis=0)
    print(f"Final input shape: {batched.shape}")

    return mx.array(batched)


def decode_predictions(predictions, model, conf_threshold=0.25):
    """Decode raw predictions to bounding boxes."""
    print("\nDecoding predictions:")
    print(f"Raw predictions shape: {predictions.shape}")

    batch_size = predictions.shape[0]
    S = model.S  # Grid size
    B = model.B  # Number of boxes per cell

    # Reshape predictions to [batch, S, S, B, 5]
    pred = mx.reshape(predictions[..., : B * 5], (batch_size, S, S, B, 5))
    mx.eval(pred)

    # Extract coordinates and confidence
    pred_xy = mx.sigmoid(pred[..., :2])  # Sigmoid for relative cell position [0,1]
    pred_wh = pred[..., 2:4]  # Raw width/height
    pred_conf = mx.sigmoid(pred[..., 4])  # Confidence score
    mx.eval(pred_xy, pred_wh, pred_conf)

    print("\nAfter sigmoid:")
    print(f"XY range: [{pred_xy.min().item():.3f}, {pred_xy.max().item():.3f}]")
    print(f"WH range: [{pred_wh.min().item():.3f}, {pred_wh.max().item():.3f}]")
    print(
        f"Confidence range: [{pred_conf.min().item():.3f}, {pred_conf.max().item():.3f}]"
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

    print("\nAfter grid transformation:")
    print(f"XY range: [{pred_xy.min().item():.3f}, {pred_xy.max().item():.3f}]")
    print(f"WH range: [{pred_wh.min().item():.3f}, {pred_wh.max().item():.3f}]")

    # Convert to corner format (x1, y1, x2, y2)
    pred_x1y1 = pred_xy - pred_wh / 2
    pred_x2y2 = pred_xy + pred_wh / 2
    boxes = mx.concatenate([pred_x1y1, pred_x2y2], axis=-1)

    # Clip to [0,1]
    boxes = mx.clip(boxes, 0.0, 1.0)

    # Reshape to [batch, S*S*B, 4]
    boxes = mx.reshape(boxes, (batch_size, S * S * B, 4))
    confidences = mx.reshape(pred_conf, (batch_size, S * S * B))
    mx.eval(boxes, confidences)

    print("\nFinal shapes:")
    print(f"Boxes: {boxes.shape}")
    print(f"Confidences: {confidences.shape}")

    return boxes, confidences


def draw_boxes(image, boxes, scores):
    """Draw predicted boxes on image"""
    draw = ImageDraw.Draw(image)

    for box, score in zip(boxes, scores):
        # Convert normalized coordinates to pixels
        x1, y1, x2, y2 = box.tolist()
        x1 = x1 * image.width
        y1 = y1 * image.height
        x2 = x2 * image.width
        y2 = y2 * image.height

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Draw confidence
        draw.text((x1, y1 - 10), f"{score:.2f}", fill="red")

    return image


def visualize_activations(frame, predictions, model):
    """Visualize activations from the final layer"""
    batch_size, channels, grid_size, _ = predictions.shape

    # Reshape predictions to [batch, S, S, B*(5+C)]
    pred = mx.reshape(predictions, (batch_size, model.S, model.S, -1))[0]
    mx.eval(pred)

    # Extract different components
    box1 = pred[..., :4]  # First box coordinates
    box2 = pred[..., 4:8]  # Second box coordinates
    class_scores = pred[..., 8:]  # Class scores

    # Create different visualizations
    vis_types = {
        "box1_xy": mx.sum(mx.abs(box1[..., :2]), axis=-1),  # Box 1 position
        "box1_wh": mx.sum(mx.abs(box1[..., 2:]), axis=-1),  # Box 1 size
        "box2_xy": mx.sum(mx.abs(box2[..., :2]), axis=-1),  # Box 2 position
        "box2_wh": mx.sum(mx.abs(box2[..., 2:]), axis=-1),  # Box 2 size
        "class": mx.sum(mx.abs(class_scores), axis=-1),  # Class confidence
    }

    # Create grid layout
    height, width = frame.shape[:2]
    cell_height = height // model.S
    cell_width = width // model.S

    # Create a layout of visualizations
    n_vis = len(vis_types)
    grid_width = 3
    grid_height = (n_vis + grid_width - 1) // grid_width

    # Create output image
    output_height = grid_height * (cell_height * model.S)
    output_width = grid_width * (cell_width * model.S)
    output = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    # Add each visualization to the grid
    for idx, (name, activations) in enumerate(vis_types.items()):
        # Convert to numpy for processing
        act_np = activations.tolist()
        act_np = np.array(act_np)

        # Normalize to [0, 1]
        act_min = np.min(act_np)
        act_max = np.max(act_np)
        if act_max > act_min:
            act_np = (act_np - act_min) / (act_max - act_min)

        # Resize to grid size
        act_resized = cv2.resize(
            act_np,
            (cell_width * model.S, cell_height * model.S),
            interpolation=cv2.INTER_NEAREST,
        )

        # Convert to heatmap
        heatmap = cv2.applyColorMap(
            (act_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        # Calculate position in grid
        grid_y = idx // grid_width
        grid_x = idx % grid_width

        # Place in output image
        y_start = grid_y * (cell_height * model.S)
        x_start = grid_x * (cell_width * model.S)
        output[
            y_start : y_start + cell_height * model.S,
            x_start : x_start + cell_width * model.S,
        ] = heatmap

        # Add label
        cv2.putText(
            output,
            name,
            (x_start + 10, y_start + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Draw grid lines
        for i in range(model.S + 1):
            # Vertical lines
            x = x_start + i * cell_width
            cv2.line(
                output,
                (x, y_start),
                (x, y_start + cell_height * model.S),
                (255, 255, 255),
                1,
            )

            # Horizontal lines
            y = y_start + i * cell_height
            cv2.line(
                output,
                (x_start, y),
                (x_start + cell_width * model.S, y),
                (255, 255, 255),
                1,
            )

    return output


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
    """Analyze a single image and show predictions vs ground truth"""
    image_path = os.path.join(args.data_dir, "JPEGImages", f"{image_id}.jpg")
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    # Load ground truth
    gt_boxes, gt_classes = load_ground_truth(image_id, args.data_dir)
    print(f"\nGround truth for {image_id}:")
    for box, cls in zip(gt_boxes, gt_classes):
        print(f"  {cls}: {box}")

    # Run inference
    input_tensor = preprocess_frame(frame)
    predictions = model(input_tensor)

    print("\nRaw predictions shape:", predictions.shape)
    print("First few values:", predictions[0, :10])  # Show first few values

    if predictions is not None:
        mx.eval(predictions)
        boxes, confidences = decode_predictions(predictions, model)
        mx.eval(boxes, confidences)

        # Convert to numpy
        boxes_np = np.array([b.tolist() for b in boxes])
        confidences_np = np.array([c.tolist() for c in confidences])
        boxes_np = boxes_np[0]  # Get first batch
        confidences_np = confidences_np[0]  # Get first batch

        print("\nDecoded boxes shape:", boxes_np.shape)
        print("Confidence scores shape:", confidences_np.shape)

        # Filter boxes
        valid_boxes, valid_confidences = filter_boxes(
            boxes_np.tolist(),
            confidences_np=(
                confidences_np.tolist() if confidences_np is not None else None
            ),
        )

        print("\nAfter filtering:")
        print(f"Number of valid boxes: {len(valid_boxes)}")
        for i, (box, conf) in enumerate(zip(valid_boxes, valid_confidences)):
            print(f"  Box {i}: {box}, confidence: {conf:.3f}")

        # Create visualization
        output_frame = frame.copy()

        # Draw ground truth in blue
        output_frame = draw_boxes(
            output_frame, gt_boxes, color=(255, 0, 0), labels=gt_classes  # Blue
        )

        # Draw predictions in confidence-based colors
        output_frame = draw_boxes(
            output_frame, valid_boxes, confidences=valid_confidences
        )

        # Add legend
        cv2.putText(
            output_frame,
            "Blue: Ground Truth, Color: Predictions (Red->Green = Low->High Confidence)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Show the image
        cv2.imshow("YOLO Detection", output_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Calculate IoU with ground truth
        print("\nIoU with ground truth:")
        for i, pred_box in enumerate(valid_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred_box, gt_box)
                print(f"  Pred {i} vs GT {j} ({gt_classes[j]}): {iou:.3f}")


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
    model = YOLO(S=7, B=5, C=20)  # B*(5+C) = 5*(5+20) = 5*25 = 125 output channels

    if not args.model.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    try:
        model.load_weights(str(args.model))
        print("Model weights loaded successfully")
        params = model.parameters()
        if not params:
            print("  Warning: No parameters loaded")
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
        # Extract image ID from path if full path provided
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
    last_boxes = None
    last_predictions = None  # Store last predictions for visualization
    show_activations = False  # Toggle for visualization mode

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

                    # Convert predictions to numpy before checking
                    if predictions is not None and predictions.tolist() is not None:
                        mx.eval(predictions)
                        last_predictions = predictions  # Store for visualization

                        boxes, confidences = decode_predictions(predictions, model)
                        mx.eval(boxes, confidences)

                        # Convert to numpy array for OpenCV
                        boxes_np = np.array([b.tolist() for b in boxes])
                        confidences_np = np.array([c.tolist() for c in confidences])
                        boxes_np = boxes_np[0]  # Get first batch
                        confidences_np = confidences_np[0]  # Get first batch

                        # Filter boxes using numpy array
                        valid_boxes, valid_confidences = filter_boxes(
                            boxes_np.tolist(),
                            confidences_np=(
                                confidences_np.tolist()
                                if confidences_np is not None
                                else None
                            ),
                        )

                        if len(valid_boxes) > 0:
                            last_boxes = valid_boxes
                            last_confidences = valid_confidences

                        # Update inference time
                        last_inference_time = current_time

                except Exception as e:
                    print(f"Error during inference: {e}")
                    import traceback

                    traceback.print_exc()
                    continue

            # Create output frame based on visualization mode
            output_frame = frame.copy()

            # Check visualization mode using Python bool operations
            if (
                show_activations
                and last_predictions is not None
                and last_predictions.tolist() is not None
            ):
                output_frame = visualize_activations(
                    output_frame, last_predictions, model
                )
            elif last_boxes is not None and len(last_boxes) > 0:
                output_frame = draw_boxes(output_frame, last_boxes, last_confidences)

            # Show the frame
            cv2.imshow("YOLO Detection", output_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("\t"):  # TAB key
                show_activations = not show_activations
                print(f"Showing {'activations' if show_activations else 'boxes'}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


def filter_boxes(boxes_np, confidences_np=None, conf_threshold=0.25):
    """Filter boxes using numpy operations"""
    min_size = 0.05  # Minimum 5% of image size
    max_size = 0.9  # Maximum 90% of image size
    max_boxes = 5  # Maximum number of boxes to show

    print("\nFiltering boxes:")
    print(f"Input boxes shape: {boxes_np}")
    if confidences_np is not None:
        print(f"Input confidences shape: {confidences_np}")

    # Convert MLX arrays to numpy if needed
    if not isinstance(boxes_np, np.ndarray):
        boxes_np = np.array(boxes_np)
    if confidences_np is not None and not isinstance(confidences_np, np.ndarray):
        confidences_np = np.array(confidences_np)

    # Calculate widths and heights
    widths = boxes_np[:, 2] - boxes_np[:, 0]
    heights = boxes_np[:, 3] - boxes_np[:, 1]
    aspect_ratios = np.maximum(widths, heights) / (np.minimum(widths, heights) + 1e-6)

    print("\nBox statistics:")
    print(f"Width range: [{widths.min():.3f}, {widths.max():.3f}]")
    print(f"Height range: [{heights.min():.3f}, {heights.max():.3f}]")
    print(f"Aspect ratio range: [{aspect_ratios.min():.3f}, {aspect_ratios.max():.3f}]")

    # Create mask for valid boxes
    valid_mask = (
        (widths > min_size)
        & (widths < max_size)
        & (heights > min_size)
        & (heights < max_size)
        & (aspect_ratios < 3)
    )

    if confidences_np is not None:
        valid_mask = valid_mask & (confidences_np > conf_threshold)

    # Get valid boxes and confidences
    valid_boxes = boxes_np[valid_mask]
    valid_confidences = (
        confidences_np[valid_mask] if confidences_np is not None else None
    )

    print(f"\nValid boxes after filtering: {len(valid_boxes)}")

    if len(valid_boxes) > 0:
        # Sort by confidence if available, otherwise by area
        if valid_confidences is not None:
            sorted_indices = np.argsort(valid_confidences)[::-1][:max_boxes]
        else:
            areas = (valid_boxes[:, 2] - valid_boxes[:, 0]) * (
                valid_boxes[:, 3] - valid_boxes[:, 1]
            )
            sorted_indices = np.argsort(areas)[::-1][:max_boxes]

        valid_boxes = valid_boxes[sorted_indices]
        valid_confidences = (
            valid_confidences[sorted_indices] if valid_confidences is not None else None
        )

        print("\nFinal boxes:")
        for i, (box, conf) in enumerate(zip(valid_boxes, valid_confidences)):
            print(f"Box {i}: {box}, confidence: {conf:.3f}")

    return valid_boxes, valid_confidences


def process_predictions(predictions, model, conf_thresh=0.2):
    """Convert raw predictions to boxes"""
    S = model.S
    B = model.B  # Now 5 boxes per cell

    # Reshape predictions to [S,S,B,5]
    predictions = mx.reshape(predictions, (S, S, B, 5))

    # Extract components
    pred_xy = mx.sigmoid(predictions[..., 0:2])  # Center coordinates
    pred_wh = predictions[..., 2:4]  # Width/height predictions
    pred_conf = mx.sigmoid(predictions[..., 4])  # Confidence scores

    # Create grid offsets
    grid_x, grid_y = mx.meshgrid(
        mx.arange(S, dtype=mx.float32), mx.arange(S, dtype=mx.float32)
    )
    grid_xy = mx.stack([grid_x, grid_y], axis=-1)  # [S,S,2]
    grid_xy = mx.expand_dims(grid_xy, axis=2)  # [S,S,1,2]

    # Convert predictions to global coordinates
    pred_xy = (pred_xy + grid_xy) / S

    # Scale width/height by anchors
    anchors = mx.expand_dims(model.anchors, axis=0)  # [1,B,2]
    anchors = mx.expand_dims(anchors, axis=0)  # [1,1,B,2]
    pred_wh = anchors * mx.exp(pred_wh) / S

    # Convert to corner format
    boxes = mx.concatenate(
        [pred_xy - pred_wh / 2, pred_xy + pred_wh / 2],  # top-left  # bottom-right
        axis=-1,
    )

    # Get confidence scores
    scores = pred_conf

    # Filter by confidence
    mask = scores > conf_thresh
    boxes = boxes[mask]
    scores = scores[mask]

    return boxes, scores


if __name__ == "__main__":
    main()
