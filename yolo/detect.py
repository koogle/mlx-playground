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
    model.load_weights(checkpoint_path)
    return model


def preprocess_image(image, size=448, args=None):
    """Preprocess image for YOLO model"""
    if isinstance(image, str):
        # Load image from file
        image = Image.open(image).convert("RGB")
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        # If OpenCV image (BGR), convert to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if args and args.debug:
        print(f"Input image shape: {image.shape}")
        print(f"Input image dtype: {image.dtype}")
        print(f"Input image range: [{image.min()}, {image.max()}]")

    # Store original size
    orig_size = image.shape[:2]  # (height, width)

    # Resize image
    image = cv2.resize(image, (size, size))

    # Convert to float and normalize
    image = image.astype(np.float32) / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Convert to MLX array
    image = mx.array(image)

    if args and args.debug:
        print(f"Preprocessed shape: {image.shape}")
        print(f"Preprocessed dtype: {image.dtype}")
        print(f"Preprocessed range: [{float(mx.min(image))}, {float(mx.max(image))}]")

    return image, orig_size


def decode_predictions(
    predictions,
    confidence_threshold=0.1,
    class_threshold=0.1,
    nms_threshold=0.4,
    debug=False,
):
    """Decode YOLO predictions to bounding boxes"""
    S = 7  # Grid size
    B = 2  # Boxes per cell
    C = len(VOC_CLASSES)  # Number of classes

    # The predictions are already in shape (batch_size, S, S, B * (5 + C))
    # No need to reshape, just split the last dimension for each box
    boxes = []
    class_ids = []
    scores = []
    confidences = []  # Store raw confidences
    class_probs = []  # Store raw class probabilities

    # For each cell in the grid
    for i in range(S):
        for j in range(S):
            # Get class probabilities (shared between boxes)
            class_offset = B * 5
            cell_class_logits = predictions[0, i, j, class_offset:class_offset + C]
            cell_class_probs = mx.softmax(cell_class_logits)  # Apply softmax to class logits

            # For each box
            for b in range(B):
                # Get box predictions
                box_offset = b * 5
                box = predictions[0, i, j, box_offset:box_offset + 5]
                
                # Apply sigmoid to confidence score
                confidence = mx.sigmoid(box[4])

                # Skip if box confidence is too low
                if float(confidence) < confidence_threshold:
                    continue

                # Get class with maximum probability
                class_id = int(mx.argmax(cell_class_probs))
                class_prob = float(cell_class_probs[class_id])

                # Skip if class probability is too low
                if class_prob < class_threshold:
                    continue

                # Convert box coordinates
                x = float((mx.sigmoid(box[0]) + j) / S)  # Add cell offset and normalize
                y = float((mx.sigmoid(box[1]) + i) / S)  # Add cell offset and normalize
                w = float(mx.exp(box[2]))  # exp for positive scaling
                h = float(mx.exp(box[3]))  # exp for positive scaling

                # Store predictions
                boxes.append([x - w/2, y - h/2, x + w/2, y + h/2])  # Convert to corners
                class_ids.append(class_id)
                scores.append(float(confidence * class_prob))
                confidences.append(float(confidence))
                class_probs.append(class_prob)

                if debug:
                    print(f"\nDetection in cell ({i}, {j}):")
                    print(f"  Class: {VOC_CLASSES[class_id]}")
                    print(f"  Box confidence: {float(confidence):.4f}")
                    print(f"  Class probability: {float(class_prob):.4f}")
                    print(
                        f"  Final score (confidence * class_prob): {float(confidence * class_prob):.4f}"
                    )
                    print(f"  Top 3 class probabilities:")
                    top_classes = [
                        int(idx) for idx in mx.argsort(cell_class_probs)[-3:][::-1]
                    ]
                    for c in top_classes:
                        print(f"    {VOC_CLASSES[c]}: {float(cell_class_probs[c]):.4f}")

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

    if len(boxes) > 0:  # Only process if there are boxes
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

    # Save or show image
    if output_path:
        image.save(output_path)
    else:
        image.show()


def draw_boxes_cv2(image, boxes, class_ids, scores):
    """Draw bounding boxes on OpenCV image with improved visualization"""
    image = image.copy()
    height, width = image.shape[:2]

    # Define a color map for different classes
    color_map = {}

    for box, class_id, score in zip(boxes, class_ids, scores):
        # Convert normalized coordinates to pixel coordinates
        x1, y1, x2, y2 = box
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)

        # Get color for this class
        if class_id not in color_map:
            color_map[class_id] = tuple(np.random.randint(0, 255, 3).tolist())
        color = color_map[class_id]

        # Draw box with thickness relative to confidence
        thickness = max(2, int(score * 4))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Prepare label with class name and score
        label = f"{VOC_CLASSES[class_id]}: {score:.2f}"

        # Get label size for background rectangle
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Draw label background
        cv2.rectangle(
            image,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            color,
            -1,
        )

        # Draw label text in white
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return image


def visualize_features(features, orig_size, window_name="Features"):
    """Visualize feature maps from the model"""
    # Convert features to numpy and take first sample from batch
    conv6 = np.array(features["conv6"][0])  # Shape: [H, W, C]
    conv7 = np.array(features["conv7"][0])  # Shape: [H, W, C]

    # Calculate mean activation across channels
    conv6_mean = np.mean(conv6, axis=-1)
    conv7_mean = np.mean(conv7, axis=-1)

    # Normalize to [0, 255] for visualization
    conv6_viz = (
        (conv6_mean - conv6_mean.min())
        / (conv6_mean.max() - conv6_mean.min() + 1e-8)
        * 255
    ).astype(np.uint8)
    conv7_viz = (
        (conv7_mean - conv7_mean.min())
        / (conv7_mean.max() - conv7_mean.min() + 1e-8)
        * 255
    ).astype(np.uint8)

    # Resize to match original image size
    height, width = orig_size
    conv6_viz = cv2.resize(conv6_viz, (width, height), interpolation=cv2.INTER_LINEAR)
    conv7_viz = cv2.resize(conv7_viz, (width, height), interpolation=cv2.INTER_LINEAR)

    # Apply colormap for better visualization
    conv6_viz = cv2.applyColorMap(conv6_viz, cv2.COLORMAP_JET)
    conv7_viz = cv2.applyColorMap(conv7_viz, cv2.COLORMAP_JET)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(conv6_viz, "Conv6", (20, 40), font, 1.0, (255, 255, 255), 2)
    cv2.putText(conv7_viz, "Conv7", (20, 40), font, 1.0, (255, 255, 255), 2)

    # Show in separate windows
    cv2.imshow(window_name + " Conv6", conv6_viz)
    cv2.imshow(window_name + " Conv7", conv7_viz)
    cv2.waitKey(1)


def debug_show_preprocessed(image):
    """Show preprocessed image for debugging"""
    # Convert MLX array to numpy and denormalize
    img = mx.astype(image[0], mx.uint8).to_numpy().astype(np.uint8) * 255

    # Convert to RGB for display
    if img.shape[-1] == 3:  # If it has 3 channels
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Create a window and show the image
    cv2.imshow("Debug: Preprocessed", img)
    cv2.waitKey(1)


def main():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--camera", action="store_true", help="Use camera input")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID")
    parser.add_argument("--output", help="Path to output image")
    parser.add_argument(
        "--conf-thresh", type=float, default=0.1, help="Confidence threshold"
    )
    parser.add_argument(
        "--class-thresh", type=float, default=0.1, help="Class probability threshold"
    )
    parser.add_argument("--nms-thresh", type=float, default=0.4, help="NMS threshold")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--show-features", action="store_true", help="Show feature map visualizations"
    )
    args = parser.parse_args()

    if not args.image and not args.camera:
        parser.error("Either --image or --camera must be specified")

    # Load model
    print("Loading model...")
    model = load_model(args.model)

    if args.camera:
        print("Opening camera... Press ESC to exit")
        cap = cv2.VideoCapture(args.camera_id)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Create windows
        cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
        if args.show_features:
            cv2.namedWindow("Feature Maps Conv6", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Feature Maps Conv7", cv2.WINDOW_NORMAL)

        process_interval = 0.5  # Process every 0.5 seconds

        try:
            while True:
                # Wait for 400ms before reading the next frame
                # This is not correct but good enough
                time.sleep(process_interval)
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                # Process frame every interval
                if args.debug:
                    print("\nProcessing new frame...")
                    print(f"Frame shape: {frame.shape}")

                # Preprocess frame
                image, orig_size = preprocess_image(frame, args=args)

                # Run inference with feature extraction
                predictions, features = model(image, return_features=True)

                # Evaluate all outputs
                mx.eval(predictions)
                mx.eval(features["conv6"])
                mx.eval(features["conv7"])

                print("Input image shape:", image.shape)
                print("Raw predictions shape:", predictions.shape)
                print("Max confidence:", float(mx.max(predictions[..., 4:5])))
                print("Max class prob:", float(mx.max(predictions[..., 10:])))
                print("Max prediction:", float(mx.max(predictions)))

                if args.debug:
                    print(
                        "Feature shapes:",
                        "conv6:",
                        features["conv6"].shape,
                        "conv7:",
                        features["conv7"].shape,
                    )
                    print(
                        "Feature ranges:",
                        "conv6:",
                        float(mx.min(features["conv6"])),
                        "to",
                        float(mx.max(features["conv6"])),
                        "conv7:",
                        float(mx.min(features["conv7"])),
                        "to",
                        float(mx.max(features["conv7"])),
                    )

                # Visualize feature maps if enabled
                if args.show_features:
                    visualize_features(features, orig_size, "Feature Maps")

                # Decode predictions
                boxes, class_ids, scores = decode_predictions(
                    predictions,
                    confidence_threshold=args.conf_thresh,
                    class_threshold=args.class_thresh,
                    nms_threshold=args.nms_thresh,
                    debug=args.debug,
                )

                # Draw results directly on the frame
                frame = draw_boxes_cv2(frame, boxes, class_ids, scores)

                # Print detections
                if len(boxes) > 0:
                    print(f"\nFound {len(boxes)} detections:")
                    for cls_id, score in zip(class_ids, scores):
                        print(f"- {VOC_CLASSES[cls_id]}: {score:.2f}")
                elif args.debug:
                    print("\nNo detections found")
                # Show the frame with detections
                cv2.imshow("YOLO Detection", frame)

                # Check for exit
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
    else:
        # Process image
        print("Processing image...")
        image, orig_size = preprocess_image(args.image, args=args)

        # Run inference with feature extraction
        predictions, features = model(image, return_features=True)

        # Evaluate all outputs
        mx.eval(predictions)
        mx.eval(features["conv6"])
        mx.eval(features["conv7"])

        # Debug prints
        if args.debug:
            print("Input image shape:", image.shape)
            print("Raw predictions shape:", predictions.shape)
            print("Max confidence:", float(mx.max(predictions[..., 4:5])))
            print("Max class prob:", float(mx.max(predictions[..., 10:])))
            print("Max prediction:", float(mx.max(predictions)))
            print(
                "Feature shapes:",
                "conv6:",
                features["conv6"].shape,
                "conv7:",
                features["conv7"].shape,
            )
            print(
                "Feature ranges:",
                "conv6:",
                float(mx.min(features["conv6"])),
                "to",
                float(mx.max(features["conv6"])),
                "conv7:",
                float(mx.min(features["conv7"])),
                "to",
                float(mx.max(features["conv7"])),
            )

        # Visualize feature maps if enabled
        if args.show_features:
            visualize_features(features, orig_size, "Feature Maps")

        # Decode predictions
        boxes, class_ids, scores = decode_predictions(
            predictions,
            args.conf_thresh,
            args.class_thresh,
            args.nms_thresh,
            debug=args.debug,
        )

        # Draw results
        print(f"Found {len(boxes)} objects")
        draw_boxes(args.image, boxes, class_ids, scores, args.output)


if __name__ == "__main__":
    main()
