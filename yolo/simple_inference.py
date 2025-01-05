import os
import cv2
import time
import mlx.core as mx
import numpy as np
from model import YOLO


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
    pred = mx.eval(pred)

    # Extract coordinates
    pred_xy = pred[..., :2]  # Center coordinates relative to cell
    pred_wh = pred[..., 2:4]  # Width and height relative to image

    # Create grid
    grid_x, grid_y = mx.meshgrid(
        mx.arange(S, dtype=mx.float32), mx.arange(S, dtype=mx.float32)
    )
    grid_xy = mx.stack([grid_x, grid_y], axis=-1)
    grid_xy = mx.expand_dims(grid_xy, axis=2)  # Add box dimension
    grid_xy = mx.eval(grid_xy)

    # Convert predictions to absolute coordinates
    cell_size = 1.0 / S
    pred_xy = mx.eval((pred_xy + grid_xy) * cell_size)  # Add cell offset and scale
    pred_wh = mx.eval(mx.clip(pred_wh, 0, 1))  # Ensure positive width/height

    # Convert to corner format (x1, y1, x2, y2)
    pred_x1y1 = mx.eval(pred_xy - pred_wh / 2)
    pred_x2y2 = mx.eval(pred_xy + pred_wh / 2)
    boxes = mx.eval(mx.concatenate([pred_x1y1, pred_x2y2], axis=-1))

    # Reshape to [batch, S*S*B, 4]
    boxes = mx.reshape(boxes, (batch_size, S * S * B, 4))
    boxes = mx.eval(boxes)

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
    # Load model
    print("Loading model...")
    model = YOLO()
    checkpoint_dir = "checkpoints"
    latest_checkpoint = max(
        [f for f in os.listdir(checkpoint_dir) if f.endswith(".safetensors")],
        key=lambda x: int(x.split("_")[2].split(".")[0]),
    )
    model_path = os.path.join(checkpoint_dir, latest_checkpoint)
    model.load_weights(model_path)
    model.eval()

    print(f"Loaded model from {model_path}")

    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    last_inference_time = 0
    inference_interval = 1.0  # Run inference every second

    print("Starting inference. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()

            # Run inference every second
            if current_time - last_inference_time >= inference_interval:
                # Preprocess frame
                input_tensor = preprocess_frame(frame)

                # Run inference and ensure all computations are complete
                predictions = model(input_tensor)
                predictions = mx.eval(predictions)

                # Decode predictions and ensure computations are complete
                boxes = decode_predictions(predictions, model)
                boxes = mx.eval(boxes)

                # Convert to numpy array for OpenCV
                boxes_np = np.array([b.tolist() for b in boxes])
                boxes_np = boxes_np[0]  # Get first batch

                # Draw boxes
                frame = draw_boxes(frame, boxes_np)

                # Update inference time
                last_inference_time = current_time

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
