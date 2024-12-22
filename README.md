# MLX Playground

This repository contains machine learning examples implemented using Apple's MLX framework, focusing on efficient implementations for Apple Silicon.

## YOLO Object Detection

A streamlined implementation of YOLO (You Only Look Once) object detection using MLX, optimized for Apple Silicon.

### Features
- YOLO v1 architecture implemented in MLX
- Training on PASCAL VOC dataset
- Memory-efficient training with gradient accumulation
- Real-time object detection capabilities
- Optimized for Apple Silicon using MLX

### Model Architecture
- Input resolution: 448x448 pixels
- Backbone: Custom CNN architecture
- Output: 7x7 grid with 2 bounding boxes per cell
- Classes: 20 PASCAL VOC classes
- Loss: Multi-part loss function (classification, localization, confidence)

### Dataset
The model is trained on PASCAL VOC 2012 with 20 object classes:
- Vehicles: aeroplane, bicycle, boat, bus, car, motorbike, train
- Animals: bird, cat, cow, dog, horse, sheep
- Indoor Objects: bottle, chair, diningtable, pottedplant, sofa, tvmonitor
- People: person

### Usage

1. Setup:
```bash
# Clone the repository
git clone https://github.com/yourusername/mlx-playground.git
cd mlx-playground

# Install dependencies
pip install mlx opencv-python pillow numpy
```

2. Training:
```bash
python yolo/train.py \
    --data-dir path/to/VOCdevkit/VOC2012 \
    --save-dir ./checkpoints \
    --batch-size 8 \
    --accumulation-steps 2
```

Training parameters:
- `--batch-size`: Number of images per mini-batch
- `--accumulation-steps`: Number of gradient accumulation steps
- `--save-dir`: Directory to save model checkpoints
- `--data-dir`: Path to VOC dataset

3. Detection:
```bash
# Image detection
python yolo/detect.py --model path/to/model.npz --image path/to/image.jpg

# Real-time camera detection
python yolo/detect.py --model path/to/model.npz --camera
```

### Implementation Details
- Efficient MLX operations for forward and backward passes
- Gradient accumulation for memory-efficient training
- Non-maximum suppression for post-processing detections
- Configurable confidence thresholds
- Real-time inference optimizations

### Project Structure
```
yolo/
├── model.py      # YOLO model architecture
├── train.py      # Training script
├── loss.py       # Loss function implementation
├── detect.py     # Detection script
└── data/         # Dataset handling
    └── voc.py    # VOC dataset loader
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests, particularly for:
- Additional ML model implementations
- Performance optimizations
- Documentation improvements
- Bug fixes

## License

MIT License