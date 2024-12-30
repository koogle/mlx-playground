"""
Verification script for YOLO model predictions.
Loads a single image and its ground truth annotations,
runs the forward pass, and analyzes the predictions in detail.
"""

import os
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import mlx.core as mx
from PIL import Image, ImageDraw

from model import YOLO
from data.voc import VOC_CLASSES
from detect import preprocess_image, decode_predictions

def load_ground_truth(annotation_path):
    """Load ground truth annotations from VOC XML file."""
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        objects.append({
            'class': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    # Get image size
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    return objects, (width, height)

def analyze_raw_predictions(predictions, S=7, B=2, C=20):
    """Analyze raw predictions from model."""
    print("\nAnalyzing Raw Predictions:")
    print(f"Shape: {predictions.shape}")
    
    # Reshape to [S, S, B*(5+C)]
    pred = mx.transpose(predictions[0], (1, 2, 0))  # [S, S, B*(5+C)]
    
    # For each cell
    for i in range(S):
        for j in range(S):
            # For each box
            for b in range(B):
                box_offset = b * 5
                class_offset = B * 5
                
                # Get box predictions
                box = pred[i, j, box_offset:box_offset + 5]
                tx, ty = mx.sigmoid(box[0:2])  # Center coordinates
                tw, th = box[2:4]  # Width/height
                conf = mx.sigmoid(box[4])  # Confidence
                
                # Get class predictions
                class_logits = pred[i, j, class_offset:class_offset + C]
                class_probs = mx.softmax(class_logits)
                top_class = mx.argmax(class_probs)
                top_prob = mx.max(class_probs)
                
                # Only print if confidence is significant
                if conf > 0.1:
                    print(f"\nCell ({i},{j}) Box {b}:")
                    print(f"  Position: tx={float(tx):.3f}, ty={float(ty):.3f}")
                    print(f"  Size: tw={float(tw):.3f}, th={float(th):.3f}")
                    print(f"  Confidence: {float(conf):.3f}")
                    print(f"  Top class: {VOC_CLASSES[int(top_class)]} ({float(top_prob):.3f})")
                    
                    # Print top 3 classes
                    top_indices = mx.argsort(class_probs)[-3:][::-1]
                    print("  Top 3 classes:")
                    for idx in top_indices:
                        prob = class_probs[idx]
                        print(f"    {VOC_CLASSES[int(idx)]}: {float(prob):.3f}")

def visualize_predictions(image_path, boxes, scores, class_ids, ground_truth=None):
    """Visualize predictions and ground truth on image."""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Draw predictions in red
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        # Convert normalized coordinates to pixel coordinates
        x1 = x1 * img.width
        y1 = y1 * img.height
        x2 = x2 * img.width
        y2 = y2 * img.height
        
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        label = f"{VOC_CLASSES[class_id]} {score:.2f}"
        draw.text((x1, y1-10), label, fill='red')
    
    # Draw ground truth in green
    if ground_truth:
        for obj in ground_truth:
            x1, y1, x2, y2 = obj['bbox']
            draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
            draw.text((x1, y1-20), obj['class'], fill='green')
    
    # Save visualization
    output_path = 'debug_output.jpg'
    img.save(output_path)
    print(f"\nVisualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Verify YOLO model predictions')
    parser.add_argument('--model', required=True, help='Path to model weights')
    parser.add_argument('--image', required=True, help='Path to test image')
    parser.add_argument('--conf-thresh', type=float, default=0.1,
                      help='Confidence threshold')
    parser.add_argument('--class-thresh', type=float, default=0.1,
                      help='Class probability threshold')
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = YOLO()
    weights = mx.load(args.model)
    model.update(weights)
    
    # Load and preprocess image
    print("Processing image...")
    image, orig_size = preprocess_image(args.image)
    
    # Get ground truth
    annotation_path = args.image.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml')
    if os.path.exists(annotation_path):
        ground_truth, image_size = load_ground_truth(annotation_path)
        print("\nGround Truth:")
        for obj in ground_truth:
            print(f"- {obj['class']} at {obj['bbox']}")
    else:
        print("\nNo ground truth annotations found")
        ground_truth = None
    
    # Run forward pass
    print("\nRunning forward pass...")
    predictions, features = model(image, return_features=True)
    
    # Analyze raw predictions
    analyze_raw_predictions(predictions)
    
    # Decode predictions
    print("\nDecoding predictions...")
    boxes, scores, class_ids = decode_predictions(
        predictions,
        confidence_threshold=args.conf_thresh,
        class_threshold=args.class_thresh
    )
    
    # Print decoded predictions
    print("\nDecoded Predictions:")
    for box, score, class_id in zip(boxes, scores, class_ids):
        print(f"- {VOC_CLASSES[class_id]}: {score:.3f} at {[float(x) for x in box]}")
    
    # Visualize results
    visualize_predictions(args.image, boxes, scores, class_ids, ground_truth)

if __name__ == '__main__':
    main()
