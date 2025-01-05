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
        # Normalize activations
        act_np = activations.tolist()
        act_np = np.array(act_np)
        act_np = (act_np - act_np.min()) / (act_np.max() - act_np.min() + 1e-6)

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
            boxes = decode_predictions(predictions, model)
            mx.eval(boxes)

            # Convert to numpy array for OpenCV
            boxes_np = np.array([b.tolist() for b in boxes])
            boxes_np = boxes_np[0]  # Get first batch

            # Filter and draw boxes
            valid_boxes = filter_boxes(boxes_np)
            output_frame = draw_boxes(frame, valid_boxes)

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


def filter_boxes(boxes_np):
    # Filter boxes based on size and aspect ratio
    min_size = 0.05  # Minimum 5% of image size
    max_size = 0.9  # Maximum 90% of image size
    max_boxes = 5  # Maximum number of boxes to show
    valid_boxes = []

    for box in boxes_np:
        w = float(box[2] - box[0])  # width
        h = float(box[3] - box[1])  # height

        if float(min_size) < w < float(max_size) and float(min_size) < h < float(
            max_size
        ):
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            if float(aspect_ratio) < 3:  # Aspect ratio less than 3:1
                valid_boxes.append(box)

    if valid_boxes:
        areas = [float((b[2] - b[0]) * (b[3] - b[1])) for b in valid_boxes]
        valid_boxes = [
            b for _, b in sorted(zip(areas, valid_boxes), reverse=True)[:max_boxes]
        ]

    return np.array(valid_boxes)


if __name__ == "__main__":
    main()
