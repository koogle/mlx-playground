# MLX Playground

This repository contains machine learning examples implemented using Apple's MLX framework, focusing on efficient implementations for Apple Silicon.

## ChessZero

A chess engine implementation inspired by AlphaZero, using MLX for neural network computations and Monte Carlo Tree Search (MCTS) for move selection.

### Key Features
- Neural MCTS implementation with extensive caching
- Highly parallelized self-play and evaluation pipeline
- Process-isolated MCTS for memory safety and parallel execution
- MLX-based neural network with residual blocks
- Self-play training with evaluation games recycling
- Bitboard-based chess engine
- Real-time move evaluation

### Architecture

#### Neural Network
- 19 residual blocks
- 256 filters per layer
- Policy head: 4672 possible moves
- Value head: Position evaluation
- Input: 19-channel board state

```
Model Configuration:
- Residual Blocks: 19
- Filters: 256
- Policy Output: 4672 moves
- Input Shape: (8, 8, 19)
- Batch Size: 2048
```

#### MCTS Implementation
- Process-isolated for memory safety
- Extensive caching system:
  - Position cache
  - Valid moves cache
  - Policy/value cache
  - Transposition table
- Early stopping with clear dominance detection
- Configurable simulation count (default 1000)
- Temperature-based exploration

#### Training Pipeline
1. Parallel self-play game generation
   - Multiple worker processes generating games simultaneously
   - Configurable number of workers for hardware optimization
   - Process isolation for memory safety and parallel execution

2. Evaluation through self-play
   - Regular evaluation against best model checkpoint
   - Parallel evaluation games for faster assessment
   - Winning evaluation games recycled into training data
   - Automatic model selection based on win rate

3. Training optimization
   - Efficient batch creation from both self-play and evaluation games
   - Regular checkpointing with best model tracking
   - Memory-optimized training process
   - Gradient updates with momentum

### Performance Features
- Multi-process game generation for both training and evaluation
- Configurable number of worker processes
- Memory-efficient process isolation
- Progress tracking for individual game workers

### Known Limitations
- Memory leaks in core MCTS implementation, fixed through process isolation
- Process isolation adds overhead

### Sample Output
```
  a b c d e f g h
8 · · · · · · · · 8
7 · · · · · ♘ · · 7
6 · · · · · · ♛ · 6
5 · · · ♙ · · · · 5
4 · ♟ · · · · ♝ ♟ 4
3 · · · · ♟ · · · 3
2 · · · ♗ · · ♚ · 2
1 ♔ · · · · · · · 1
  a b c d e f g h
```

### ChessZero Usage

1. Training with parallel processing:
```bash
python chesszero/train.py --workers 8  # Adjust worker count based on CPU cores
```

2. Play against AI:
```bash
python chesszero/chess_engine/main.py --mode ai
```

3. Watch AI self-play:
```bash
python chesszero/chess_engine/main.py --mode auto
```

## YOLO Object Detection

A streamlined implementation of YOLOv2 (You Only Look Once v2) object detection using MLX, optimized for Apple Silicon.

### Features
- YOLOv2 architecture with anchor boxes
- Training on PASCAL VOC dataset
- Memory-efficient training
- Real-time object detection capabilities
- Optimized for Apple Silicon using MLX

### Model Architecture
- Input resolution: 448x448 pixels
- Backbone: darknet-19
- Output: 7x7 grid with 5 anchor boxes per cell
- Anchor box predictions: (x, y, w, h, confidence)
- Classes: 20 PASCAL VOC classes
- Loss: Multi-part loss function (classification, localization, confidence)

### Training Modes
1. Development Mode:
```bash
python yolo/simple_train.py --mode dev
```
- Uses small subset of data (10 images)
- Smaller batch size
- Faster iteration for testing changes

2. Full Training:
```bash
python yolo/simple_train.py --mode full
```
- Uses complete dataset
- Larger batch size
- Regular validation and checkpointing

### Dataset
The model is trained on PASCAL VOC 2012 with 20 object classes:
- Vehicles: aeroplane, bicycle, boat, bus, car, motorbike, train
- Animals: bird, cat, cow, dog, horse, sheep
- Indoor Objects: bottle, chair, diningtable, pottedplant, sofa, tvmonitor
- People: person

### YOLO Usage

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

Contributions are welcome! Key areas for improvement:
- MCTS memory management
- Evaluation function
- Performance optimizations
- Documentation improvements
- Bug fixes

## License

MIT License
