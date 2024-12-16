import os
import argparse
import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw
import cv2
from model import YOLO
from data.voc import VOC_CLASSES
import time


def load_model(checkpoint_path):
    """Load trained model from checkpoint"""
    model = YOLO()
    state_dict = mx.load(checkpoint_path)
    model.update(state_dict)
    return model


def preprocess_image(image, size=448):
    """Load and preprocess image for inference"""
    if isinstance(image, str):
        # If input is a path, load the image
        image = Image.open(image).convert("RGB")
        orig_size = image.size
    else:
        # If input is a numpy array (from cv2), convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_size = (image.shape[1], image.shape[0])
        image = Image.fromarray(image)

    # Resize and convert to numpy array
    image = image.resize((size, size))
    image = np.array(image, dtype=np.float32) / 255.0

    # Convert to MLX array (keeping NHWC format)
    image = mx.array(image)
    # Add batch dimension
    image = mx.expand_dims(image, axis=0)

    return image, orig_size


def decode_predictions(predictions, confidence_threshold=0.5, nms_threshold=0.4):
    """Decode YOLO predictions to bounding boxes"""
    S = 7  # Grid size
    B = 2  # Boxes per cell
    C = 20  # Number of classes

    # Reshape predictions
    predictions = predictions.reshape(-1, S, S, B * 5 + C)

    boxes = []
    class_ids = []
    scores = []

    # For each cell in the grid
    for i in range(S):
        for j in range(S):
            # Get class probabilities
            class_probs = predictions[0, i, j, B * 5 :]

            # For each box
            for b in range(B):
                # Get box predictions
                box = predictions[0, i, j, b * 5 : (b + 1) * 5]
                confidence = box[4]

                # Get class with maximum probability
                class_id = mx.argmax(class_probs)
                class_prob = class_probs[class_id]

                # Final score
                score = confidence * class_prob

                if score > confidence_threshold:
                    # Convert box coordinates
                    x = (box[0] + i) / S
                    y = (box[1] + j) / S
                    w = box[2]
                    h = box[3]

                    # Convert to corner format
                    x1 = x - w / 2
                    y1 = y - h / 2
                    x2 = x + w / 2
                    y2 = y + h / 2

                    boxes.append([x1, y1, x2, y2])
                    class_ids.append(class_id)
                    scores.append(score)

    if not boxes:
        return [], [], []

    # Convert to numpy for NMS
    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)

    # Apply NMS
    selected_indices = []
    for class_id in np.unique(class_ids):
        class_mask = class_ids == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]

        while len(class_boxes) > 0:
            max_idx = np.argmax(class_scores)
            selected_indices.append(np.where(class_mask)[0][max_idx])

            # Compute IoU with remaining boxes
            ious = compute_iou_numpy(class_boxes[max_idx], class_boxes)
            mask = ious <= nms_threshold

            class_boxes = class_boxes[mask]
            class_scores = class_scores[mask]

    return (
        boxes[selected_indices],
        class_ids[selected_indices],
        scores[selected_indices],
    )


def compute_iou_numpy(box, boxes):
    """Compute IoU between a box and an array of boxes"""
    # Calculate intersection
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Calculate union
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection

    return intersection / (union + 1e-6)


def draw_boxes(image_path, boxes, class_ids, scores, output_path=None):
    """Draw bounding boxes on image"""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Scale boxes to image size
    width, height = image.size
    boxes = boxes * np.array([width, height, width, height])

    for box, class_id, score in zip(boxes, class_ids, scores):
        x1, y1, x2, y2 = box

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Draw label
        label = f"{VOC_CLASSES[class_id]}: {score:.2f}"
        draw.text((x1, y1 - 10), label, fill="red")

    if output_path:
        image.save(output_path)
    else:
        image.show()


def draw_boxes_cv2(image, boxes, class_ids, scores):
    """Draw bounding boxes on OpenCV image"""
    # Return original image if no detections
    if len(boxes) == 0 or boxes.size == 0:
        return image

    height, width = image.shape[:2]
    boxes = boxes * np.array([width, height, width, height])

    for box, class_id, score in zip(boxes, class_ids, scores):
        x1, y1, x2, y2 = map(int, box)

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw label
        label = f"{VOC_CLASSES[class_id]}: {score:.2f}"
        cv2.putText(
            image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )

    return image


def main():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--camera", action="store_true", help="Use camera input")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID")
    parser.add_argument("--output", help="Path to output image")
    parser.add_argument(
        "--conf-thresh", type=float, default=0.5, help="Confidence threshold"
    )
    parser.add_argument("--nms-thresh", type=float, default=0.4, help="NMS threshold")
    args = parser.parse_args()

    if not args.image and not args.camera:
        parser.error("Either --image or --camera must be specified")

    # Load model
    print("Loading model...")
    model = load_model(args.model)

    if args.camera:
        print("Opening camera...")
        cap = cv2.VideoCapture(args.camera_id)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size

        fps = 0
        frame_count = 0
        start_time = time.time()
        last_inference_time = time.time()
        last_boxes, last_class_ids, last_scores = [], [], []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                # Update FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    end_time = time.time()
                    fps = 30 / (end_time - start_time)
                    start_time = time.time()

                # Run inference only once per second
                current_time = time.time()
                if current_time - last_inference_time >= 1.0:
                    # Preprocess frame
                    image, orig_size = preprocess_image(frame)

                    # Run inference
                    predictions = model(image)
                    mx.eval(predictions)

                    # Decode predictions
                    last_boxes, last_class_ids, last_scores = decode_predictions(
                        predictions,
                        confidence_threshold=args.conf_thresh,
                        nms_threshold=args.nms_thresh,
                    )
                    last_inference_time = current_time

                    # Print detections to console
                    if len(last_boxes) > 0:
                        print("\nDetections:")
                        for cls_id, score in zip(last_class_ids, last_scores):
                            print(f"- {VOC_CLASSES[cls_id]}: {score:.2f}")
                    else:
                        print("\nNo detections")

                # Draw results using last available predictions
                frame = draw_boxes_cv2(frame, last_boxes, last_class_ids, last_scores)

                # Add FPS counter and inference info
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Inference: {time.time() - last_inference_time:.1f}s ago",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Add detection count
                status = (
                    f"Detected: {len(last_boxes)} objects"
                    if len(last_boxes) > 0
                    else "No detections"
                )
                cv2.putText(
                    frame,
                    status,
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Display the frame
                cv2.imshow("YOLO Detection", frame)
                cv2.waitKey(1)  # This is crucial for window updates

                # Break on 'q' press
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except Exception as e:
            print(f"Error during camera capture: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    else:
        # Process single image
        print("Processing image...")
        image, orig_size = preprocess_image(args.image)

        # Run inference
        predictions = model(image)
        mx.eval(predictions)

        # Decode predictions
        boxes, class_ids, scores = decode_predictions(
            predictions,
            confidence_threshold=args.conf_thresh,
            nms_threshold=args.nms_thresh,
        )

        # Draw results
        print(f"Found {len(boxes)} objects")
        draw_boxes(args.image, boxes, class_ids, scores, args.output)


if __name__ == "__main__":
    main()
