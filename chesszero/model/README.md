# Neural Network Model

This directory contains the neural network and learning components of the ChessZero project.

## Components

### Neural Network (`network.py`)
- MLX-based implementation of the AlphaZero-style neural network
- Architecture:
  - 19 residual blocks
  - 256 filters per convolutional layer
  - Policy head: outputs probabilities for 4672 possible moves
  - Value head: evaluates board positions
- Input: 19-channel board state representation (8×8×19)
- Training with Adam optimizer

### Monte Carlo Tree Search (`mcts.py`)
- Policy-guided MCTS implementation
- Uses neural network for position evaluation and move probability
- Extensive caching system:
  - Position cache
  - Valid moves cache
  - Policy/value prediction cache
  - Transposition table
- Early stopping with clear dominance detection
- Temperature-based exploration
- Parallel execution support

### Self-Play (`self_play.py`)
- Generates training data through self-play
- Multiprocessing for parallel game generation
- Process isolation for memory safety
- Converts game results into training examples
- Creates batches for efficient training
- Game result statistics tracking

### Training (`training.py`)
- Implements the full training loop
- Manages:
  - Self-play data generation
  - Neural network updates
  - Model evaluation
  - Checkpoint saving/loading
- Automated model selection based on evaluation performance
- Training metrics tracking

## Integration

The components work together in a reinforcement learning loop:
1. The neural network guides MCTS to make strong moves
2. Self-play generates training data using MCTS
3. The network is trained on this data to improve
4. The improved network is evaluated against previous versions
5. The cycle continues, leading to progressively stronger play