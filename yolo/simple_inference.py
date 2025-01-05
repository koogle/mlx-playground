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
    min_size = 0.01  # Minimum 1% of image size
    pred_wh = mx.clip(pred_wh, min_size, 1.0)  # Ensure reasonable size
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
    last_boxes = None  # Store last valid boxes

    print("Starting inference. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()

            # Run inference every interval
            if current_time - last_inference_time >= inference_interval:
                try:
                    # Preprocess frame
                    input_tensor = preprocess_frame(frame)
                    print(f"\nInput shape: {input_tensor.shape}")

                    # Run inference with debug info
                    try:
                        predictions = model(input_tensor)
                        print("Forward pass completed")
                    except Exception as e:
                        print(f"Error in model forward pass: {e}")
                        print("Model state:")
                        for name, param in model.parameters().items():
                            print(f"  {name}: {param.shape}")
                        continue

                    if predictions is None:
                        print("Error: Model returned None predictions")
                        continue

                    # Ensure predictions are evaluated
                    mx.eval(predictions)
                    print(f"Predictions shape: {predictions.shape}")

                    # Decode predictions
                    boxes = decode_predictions(predictions, model)
                    if boxes is None:
                        print("Error: Failed to decode predictions")
                        continue

                    # Ensure boxes are evaluated
                    mx.eval(boxes)
                    print(f"Boxes shape: {boxes.shape}")

                    # Convert to numpy array for OpenCV
                    boxes_np = np.array([b.tolist() for b in boxes])
                    boxes_np = boxes_np[0]  # Get first batch

                    # Filter boxes based on size threshold
                    min_size = 0.05  # Minimum 5% of image size
                    valid_boxes = []
                    for box in boxes_np:
                        w = box[2] - box[0]  # width
                        h = box[3] - box[1]  # height
                        if w > min_size and h > min_size:
                            valid_boxes.append(box)

                    if valid_boxes:
                        last_boxes = np.array(valid_boxes)
                        print(f"Found {len(valid_boxes)} valid boxes")

                    # Update inference time
                    last_inference_time = current_time

                except Exception as e:
                    print(f"Error during inference: {e}")
                    import traceback

                    traceback.print_exc()
                    continue

            # Draw boxes (using last valid boxes)
            if last_boxes is not None:
                frame = draw_boxes(frame, last_boxes)

            # Show the frame
            cv2.imshow("YOLO Detection", frame)

            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
