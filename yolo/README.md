# YOLO Object Detection

A streamlined implementation of YOLOv2 (You Only Look Once v2) object detection using MLX, optimized for Apple Silicon.

## Features
- YOLOv2 architecture with anchor boxes
- Training on PASCAL VOC dataset
- Memory-efficient training
- Real-time object detection capabilities
- Optimized for Apple Silicon using MLX

## Model Architecture
- Input resolution: 448x448 pixels
- Backbone: darknet-19
- Output: 7x7 grid with 5 anchor boxes per cell
- Anchor box predictions: (x, y, w, h, confidence)
- Classes: 20 PASCAL VOC classes
- Loss: Multi-part loss function (classification, localization, confidence)

## Components
- `model.py`: YOLO model architecture
- `loss.py`: Loss function implementation
- `simple_train.py`: Training script
- `simple_inference.py`: Inference script
- `compute_anchors.py`: Utility for computing anchor boxes
- `data/`: Dataset handling utilities

## Training Modes
1. Development Mode:
```bash
python simple_train.py --mode dev
```
- Uses small subset of data (10 images)
- Smaller batch size
- Faster iteration for testing changes

2. Full Training:
```bash
python simple_train.py --mode full
```
- Uses complete dataset
- Larger batch size
- Regular validation and checkpointing

## Dataset
The model is trained on PASCAL VOC 2012 with 20 object classes:
- Vehicles: aeroplane, bicycle, boat, bus, car, motorbike, train
- Animals: bird, cat, cow, dog, horse, sheep
- Indoor Objects: bottle, chair, diningtable, pottedplant, sofa, tvmonitor
- People: person