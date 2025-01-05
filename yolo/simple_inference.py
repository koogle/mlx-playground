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
    Returns boxes in (x1, y1, x2, y2) format.
    """
    batch_size = predictions.shape[0]
    S = model.S  # Grid size
    B = model.B  # Number of boxes per cell

    # Reshape predictions to [batch, S, S, B, 4]
    pred = mx.reshape(predictions[..., : B * 4], (batch_size, S, S, B, 4))
    mx.eval(pred)

    # Extract coordinates
    pred_xy = pred[..., :2]  # Center coordinates relative to cell
    pred_wh = pred[..., 2:4]  # Width and height relative to image

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

    # Clip and threshold width/height predictions
    min_size = 0.05  # Increased minimum size to 5%
    max_size = 0.9  # Maximum 90% of image size
    pred_wh = mx.clip(pred_wh, min_size, max_size)  # Ensure reasonable size
    mx.eval(pred_xy, pred_wh)

    # Convert to corner format (x1, y1, x2, y2)
    pred_x1y1 = pred_xy - pred_wh / 2
    pred_x2y2 = pred_xy + pred_wh / 2
    boxes = mx.concatenate([pred_x1y1, pred_x2y2], axis=-1)
    mx.eval(boxes)

    # Clip boxes to image boundaries
    boxes = mx.clip(boxes, 0.0, 1.0)
    mx.eval(boxes)

    # Reshape to [batch, S*S*B, 4]
    boxes = mx.reshape(boxes, (batch_size, S * S * B, 4))
    mx.eval(boxes)

    return boxes


def draw_boxes(frame, boxes, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on frame"""
    height, width = frame.shape[:2]

    for box in boxes:
        # Convert normalized coordinates to pixel coordinates
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    return frame


def visualize_activations(frame, predictions, model):
    """Visualize activations from the final layer"""
    batch_size, channels, grid_size, _ = predictions.shape

    # Sum activations across channels to get activation strength per grid cell
    activations = mx.sum(mx.abs(predictions), axis=1)  # Shape: [batch, S, S]
    activations = activations[0]  # Take first batch
    mx.eval(activations)

    # Convert to numpy and normalize to [0, 1]
    act_np = activations.tolist()
    act_np = np.array(act_np)
    act_np = (act_np - act_np.min()) / (act_np.max() - act_np.min() + 1e-6)

    # Resize to frame size
    height, width = frame.shape[:2]
    act_resized = cv2.resize(act_np, (width, height), interpolation=cv2.INTER_LINEAR)

    # Convert to heatmap
    heatmap = cv2.applyColorMap((act_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Blend with original frame
    alpha = 0.7
    return cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)


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
        print("\nModel parameters:")
        params = model.parameters()
        if not params:
            print("  Warning: No parameters loaded")
        else:
            for name, param in params.items():
                if isinstance(param, dict):
                    print(f"  {name}:")
                    for k, v in param.items():
                        print(
                            f"    {k}: {v.shape if hasattr(v, 'shape') else 'no shape'}"
                        )
                else:
                    print(
                        f"  {name}: {param.shape if hasattr(param, 'shape') else 'no shape'}"
                    )

        # Print model structure
        print("\nModel structure:")
        print(model)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        import traceback

        traceback.print_exc()
        return

    model.eval()
    print(f"Loaded model from {args.model}")

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
                    if predictions is None:
                        continue

                    mx.eval(predictions)
                    last_predictions = predictions  # Store for visualization

                    boxes = decode_predictions(predictions, model)
                    mx.eval(boxes)

                    # Convert to numpy array for OpenCV
                    boxes_np = np.array([b.tolist() for b in boxes])
                    boxes_np = boxes_np[0]  # Get first batch

                    # Filter boxes based on size and aspect ratio
                    min_size = 0.05  # Minimum 5% of image size
                    max_size = 0.9  # Maximum 90% of image size
                    max_boxes = 5  # Maximum number of boxes to show
                    valid_boxes = []

                    for box in boxes_np:
                        w = float(box[2] - box[0])  # width
                        h = float(box[3] - box[1])  # height

                        # Convert to Python floats for comparison
                        if float(min_size) < w < float(max_size) and float(
                            min_size
                        ) < h < float(max_size):
                            # Calculate aspect ratio safely
                            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
                            if float(aspect_ratio) < 3:  # Aspect ratio less than 3:1
                                valid_boxes.append(box)

                    # Keep only the largest boxes up to max_boxes
                    if valid_boxes:  # Python list check
                        areas = [
                            float((b[2] - b[0]) * (b[3] - b[1])) for b in valid_boxes
                        ]
                        valid_boxes = [
                            b
                            for _, b in sorted(zip(areas, valid_boxes), reverse=True)[
                                :max_boxes
                            ]
                        ]
                        last_boxes = np.array(valid_boxes)

                    # Update inference time
                    last_inference_time = current_time

                except Exception as e:
                    print(f"Error during inference: {e}")
                    continue

            # Create output frame based on visualization mode
            output_frame = frame.copy()

            if show_activations and last_predictions is not None:
                output_frame = visualize_activations(
                    output_frame, last_predictions, model
                )
            elif last_boxes is not None and len(last_boxes) > 0:
                output_frame = draw_boxes(output_frame, last_boxes)

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


if __name__ == "__main__":
    main()
