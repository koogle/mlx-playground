# MLX Playground

This repository contains various machine learning examples implemented using Apple's MLX framework. It serves as a playground for experimenting with MLX's features and capabilities.

## Examples

### 1. MNIST Classification

A simple implementation of MNIST digit classification using MLX.

#### Features
- Basic CNN model for digit classification
- MNIST dataset loading and preprocessing
- Training and evaluation scripts

#### Usage
```bash
# Train the MNIST model
python mnist/train.py

# Test the model
python mnist/test.py
```

### 2. YOLO Object Detection

Implementation of YOLO (You Only Look Once) object detection using MLX.

#### Features
- YOLO model implementation in MLX
- Real-time object detection using webcam
- Support for image file detection
- Training on PASCAL VOC dataset
- Memory-optimized training with gradient accumulation

#### Supported Classes
The YOLO model can detect 20 object classes from PASCAL VOC:
- Vehicles: aeroplane, bicycle, boat, bus, car, motorbike, train
- Animals: bird, cat, cow, dog, horse, sheep
- Indoor Objects: bottle, chair, diningtable, pottedplant, sofa, tvmonitor
- People: person

#### Usage

1. Training:
```bash
python yolo/train.py \
    --data-dir path/to/VOCdevkit/VOC2012 \
    --save-dir ./checkpoints \
    --batch-size 8 \
    --accumulation-steps 2
```

2. Detection:
```bash
# Image detection
python yolo/detect.py --model path/to/model.npz --image path/to/image.jpg

# Real-time camera detection
python yolo/detect.py --model path/to/model.npz --camera
```

Camera Controls:
- Press 'q' to quit
- Detection runs once per second for optimal performance
- On-screen display shows FPS, detection count, and inference timing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mlx-playground.git
cd mlx-playground
```

2. Install dependencies:
```bash
pip install mlx opencv-python pillow numpy
```

## Implementation Details

### MNIST
- Simple CNN architecture
- Input: 28x28 grayscale images
- Output: 10 classes (digits 0-9)
- Dataset: Standard MNIST dataset

### YOLO
- Architecture: YOLO v1
- Input Size: 448x448 pixels
- Grid Size: 7x7
- Boxes per Cell: 2
- Dataset: PASCAL VOC 2012
- Optimizations:
  - Gradient accumulation for memory-efficient training
  - Once-per-second inference for real-time detection
  - Configurable confidence threshold
  - Non-maximum suppression for overlapping detections

## Contributing

Feel free to contribute additional examples or improvements to existing ones. Please follow the existing code structure and add appropriate documentation.

## License

[Your License Here]