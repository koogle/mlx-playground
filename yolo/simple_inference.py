import os
import cv2
import time
import mlx.core as mx
import numpy as np
from model import YOLO
import argparse
from pathlib import Path


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
        help="Path to an image file for inference",
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

    # Convert to MLX array
    return mx.array(batched)


def decode_predictions(predictions, model, conf_threshold=0.25):
    """
    Decode raw predictions to bounding boxes.
    Returns boxes in (x1, y1, x2, y2) format and confidence scores.
    """
    batch_size = predictions.shape[0]
    S = model.S  # Grid size
    B = model.B  # Number of boxes per cell

    # Reshape predictions to [batch, S, S, B, 5]
    pred = mx.reshape(predictions[..., : B * 5], (batch_size, S, S, B, 5))
    mx.eval(pred)

    # Extract coordinates and confidence
    pred_xy = pred[..., :2]  # Center coordinates relative to cell
    pred_wh = pred[..., 2:4]  # Width and height relative to image
    pred_conf = mx.sigmoid(pred[..., 4])  # Confidence score
    mx.eval(pred_conf)

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
    pred_wh = mx.clip(pred_wh, 0.01, 1.0)  # Ensure reasonable size
    mx.eval(pred_xy, pred_wh)

    # Convert to corner format (x1, y1, x2, y2)
    pred_x1y1 = pred_xy - pred_wh / 2
    pred_x2y2 = pred_xy + pred_wh / 2
    boxes = mx.concatenate([pred_x1y1, pred_x2y2], axis=-1)
    mx.eval(boxes)

    # Reshape boxes and confidence scores
    boxes = mx.reshape(boxes, (batch_size, S * S * B, 4))
    confidences = mx.reshape(pred_conf, (batch_size, S * S * B))
    mx.eval(boxes, confidences)

    return boxes, confidences


def draw_boxes(frame, boxes, confidences=None, thickness=2):
    """Draw bounding boxes on frame with color based on confidence"""
    height, width = frame.shape[:2]

    for i, box in enumerate(boxes):
        # Convert normalized coordinates to pixel coordinates
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        # Color based on confidence (red->yellow->green)
        if confidences is not None:
            conf = confidences[i]
            # Red channel decreases with confidence
            r = int(255 * (1 - conf))
            # Green channel increases with confidence
            g = int(255 * conf)
            color = (0, g, r)  # BGR format
        else:
            color = (0, 255, 0)  # Default green

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Add confidence text if available
        if confidences is not None:
            conf_text = f"{conf:.2f}"
            cv2.putText(
                frame,
                conf_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    return frame


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


def main():
    args = parse_args()

    # Load model
    print("Loading model...")
    model = YOLO()

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
        import traceback

        traceback.print_exc()
        return

    model.eval()
    print(f"Loaded model from {args.model}")

    if args.image:
        # Perform inference on a single image
        if not args.image.exists():
            raise FileNotFoundError(f"Image not found: {args.image}")

        frame = cv2.imread(str(args.image))
        if frame is None:
            raise RuntimeError(f"Failed to load image: {args.image}")

        input_tensor = preprocess_frame(frame)
        predictions = model(input_tensor)
        if predictions is not None:
            mx.eval(predictions)
            boxes, confidences = decode_predictions(predictions, model)
            mx.eval(boxes, confidences)

            # Convert to numpy array for OpenCV
            boxes_np = np.array([b.tolist() for b in boxes])
            boxes_np = boxes_np[0]  # Get first batch

            # Filter and draw boxes
            valid_boxes, valid_confidences = filter_boxes(
                boxes_np.tolist(),
                confidences_np=(
                    confidences[0].tolist() if confidences is not None else None
                ),
            )
            output_frame = draw_boxes(frame, valid_boxes, confidences=valid_confidences)

            # Show the image with detections
            cv2.imshow("YOLO Detection", output_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
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

    # Convert MLX arrays to numpy if needed
    if not isinstance(boxes_np, np.ndarray):
        boxes_np = np.array(boxes_np.tolist())
    if confidences_np is not None and not isinstance(confidences_np, np.ndarray):
        confidences_np = np.array(confidences_np.tolist())

    # Calculate widths and heights
    widths = boxes_np[:, 2] - boxes_np[:, 0]
    heights = boxes_np[:, 3] - boxes_np[:, 1]
    aspect_ratios = np.maximum(widths, heights) / (np.minimum(widths, heights) + 1e-6)

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

    return valid_boxes, valid_confidences


if __name__ == "__main__":
    main()
