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
    """
    Preprocess image for YOLO model

    Returns image in NHWC format
    """
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

    # Add batch dimension (keeping NHWC format)
    image = np.expand_dims(image, axis=0)  # [1, H, W, C]

    # Convert to MLX array
    image = mx.array(image)

    if args and args.debug:
        print(f"Preprocessed shape: {image.shape}")
        print(f"Preprocessed dtype: {image.dtype}")
        print(f"Preprocessed range: [{float(mx.min(image))}, {float(mx.max(image))}]")

    return image, orig_size


def process_predictions(predictions, model, conf_threshold=0.25, nms_threshold=0.45):
    """
    Process raw model predictions to get final detections.
    """
    # Convert predictions from NCHW to NHWC format
    predictions = mx.transpose(predictions, (0, 2, 3, 1))
    
    # Extract dimensions
    batch_size = predictions.shape[0]
    S = model.S
    B = model.B
    C = model.C
    
    # Reshape predictions
    predictions = mx.reshape(predictions, (batch_size, S, S, B, 5 + C))
    
    # Extract components with numerical stability
    pred_xy = mx.sigmoid(mx.clip(predictions[..., 0:2], -10, 10))
    pred_wh = mx.clip(predictions[..., 2:4], -10, 10)
    pred_conf = mx.sigmoid(mx.clip(predictions[..., 4:5], -10, 10))
    pred_class = predictions[..., 5:]
    
    # Apply softmax to class predictions with numerical stability
    pred_class = pred_class - mx.max(pred_class, axis=-1, keepdims=True)
    pred_class = mx.exp(pred_class)
    pred_class = pred_class / (mx.sum(pred_class, axis=-1, keepdims=True) + 1e-10)
    
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
    
    # Convert to corner format (x1, y1, x2, y2)
    pred_mins = pred_xy_abs - pred_wh_abs / 2.0
    pred_maxs = pred_xy_abs + pred_wh_abs / 2.0
    pred_boxes = mx.concatenate([pred_mins, pred_maxs], axis=-1)
    
    # Get class scores and ids
    class_scores = mx.max(pred_class, axis=-1)
    class_ids = mx.argmax(pred_class, axis=-1)
    
    # Confidence scores
    scores = pred_conf[..., 0] * class_scores
    
    # Filter by confidence threshold
    mask = scores > conf_threshold
    
    # Collect all detections
    all_boxes = []
    all_scores = []
    all_classes = []
    
    for b in range(batch_size):
        # Get boxes for this image
        boxes = pred_boxes[b][mask[b]]
        det_scores = scores[b][mask[b]]
        det_classes = class_ids[b][mask[b]]
        
        if len(boxes) == 0:
            all_boxes.append(mx.array([]))
            all_scores.append(mx.array([]))
            all_classes.append(mx.array([]))
            continue
        
        # Apply NMS per class
        final_boxes = []
        final_scores = []
        final_classes = []
        
        for class_id in range(C):
            class_mask = det_classes == class_id
            if not mx.any(class_mask):
                continue
            
            c_boxes = boxes[class_mask]
            c_scores = det_scores[class_mask]
            
            # Sort by score
            indices = mx.argsort(c_scores, descending=True)
            c_boxes = c_boxes[indices]
            c_scores = c_scores[indices]
            
            # Apply NMS
            keep = []
            while len(c_boxes) > 0:
                keep.append(0)
                if len(c_boxes) == 1:
                    break
                    
                # Compute IoU with remaining boxes
                ious = compute_box_iou(
                    c_boxes[0:1],
                    c_boxes[1:]
                )
                
                # Filter boxes with IoU > threshold
                mask = ious <= nms_threshold
                c_boxes = mx.concatenate([c_boxes[0:1], c_boxes[1:][mask]])
                c_scores = mx.concatenate([c_scores[0:1], c_scores[1:][mask]])
            
            final_boxes.extend(c_boxes[keep])
            final_scores.extend(c_scores[keep])
            final_classes.extend([class_id] * len(keep))
        
        all_boxes.append(mx.stack(final_boxes) if final_boxes else mx.array([]))
        all_scores.append(mx.stack(final_scores) if final_scores else mx.array([]))
        all_classes.append(mx.stack(final_classes) if final_classes else mx.array([]))
    
    return all_boxes, all_scores, all_classes


def detect_objects(model, image_path, conf_threshold=0.25, nms_threshold=0.45):
    """
    Detect objects in an image using the YOLO model.
    """
    # Load and preprocess image
    image = load_image(image_path)
    input_tensor = preprocess_image(image)
    
    # Run model
    predictions = model(input_tensor)
    
    # Process predictions
    boxes, scores, class_ids = process_predictions(
        predictions,
        model,
        conf_threshold,
        nms_threshold
    )
    
    return boxes[0], scores[0], class_ids[0]  # Return results for first image


def compute_box_iou(box1, box2):
    """Compute IoU between two boxes"""
    # Calculate intersection
    x1 = mx.maximum(box1[:, 0], box2[:, 0])
    y1 = mx.maximum(box1[:, 1], box2[:, 1])
    x2 = mx.minimum(box1[:, 2], box2[:, 2])
    y2 = mx.minimum(box1[:, 3], box2[:, 3])

    intersection = mx.maximum(0, x2 - x1) * mx.maximum(0, y2 - y1)

    # Calculate union
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = box1_area + box2_area - intersection

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
    parser = argparse.ArgumentParser(description="YOLO object detection")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--video", help="Path to input video")
    parser.add_argument("--camera", action="store_true", help="Use camera input")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID")
    parser.add_argument("--output", help="Path to output image")
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.25,  # Slightly increased
        help="Confidence threshold (objectness score)",
    )
    parser.add_argument(
        "--class-thresh",
        type=float,
        default=0.25,       # Slightly increased
        help="Class probability threshold",
    )
    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=0.45,         # Relaxed a bit
        help="NMS IoU threshold (lower = stricter filtering of overlapping boxes)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--show-features", action="store_true", help="Show feature map visualizations"
    )
    args = parser.parse_args()

    if not args.image and not args.video and not args.camera:
        parser.error("Either --image, --video or --camera must be specified")

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

                # Process predictions
                boxes, scores, class_ids = process_predictions(
                    predictions,
                    model,
                    conf_threshold=args.conf_thresh,
                    nms_threshold=args.nms_thresh,
                )

                # Draw results directly on the frame
                frame = draw_boxes_cv2(frame, boxes[0], class_ids[0], scores[0])

                # Print detections
                if len(boxes[0]) > 0:
                    print(f"\nFound {len(boxes[0])} detections:")
                    for cls_id, score in zip(class_ids[0], scores[0]):
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
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print("Error: Could not open video")
            return

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

                # Process predictions
                boxes, scores, class_ids = process_predictions(
                    predictions,
                    model,
                    conf_threshold=args.conf_thresh,
                    nms_threshold=args.nms_thresh,
                )

                # Draw results directly on the frame
                frame = draw_boxes_cv2(frame, boxes[0], class_ids[0], scores[0])

                # Print detections
                if len(boxes[0]) > 0:
                    print(f"\nFound {len(boxes[0])} detections:")
                    for cls_id, score in zip(class_ids[0], scores[0]):
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

        # Process predictions
        boxes, scores, class_ids = process_predictions(
            predictions,
            model,
            conf_threshold=args.conf_thresh,
            nms_threshold=args.nms_thresh,
        )

        # Draw results
        print(f"Found {len(boxes[0])} objects")
        draw_boxes(args.image, boxes[0], class_ids[0], scores[0], args.output)


if __name__ == "__main__":
    main()
