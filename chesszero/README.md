# ChessZero

A chess engine implementation inspired by AlphaZero, using MLX for neural network computations and Monte Carlo Tree Search (MCTS) for move selection.

## Key Features
- Neural MCTS implementation with extensive caching
- Highly parallelized self-play and evaluation pipeline
- Process-isolated MCTS for memory safety and parallel execution
- MLX-based neural network with residual blocks
- Automated model selection through competitive self-play evaluation
- Bitboard-based chess engine
- Real-time move evaluation

## Architecture

### Neural Network
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

### MCTS Implementation
- Process-isolated for memory safety
- Extensive caching system:
  - Position cache
  - Valid moves cache
  - Policy/value cache
  - Transposition table
- Early stopping with clear dominance detection
- Configurable simulation count (default 1000)
- Temperature-based exploration

### Training Pipeline
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
   - Using Adam optimizer

## Performance Features
- Multi-process game generation for both training and evaluation
- Configurable number of worker processes
- Memory-efficient process isolation
- Progress tracking for individual game workers

## Known Limitations
- Memory leaks in core MCTS implementation, fixed through process isolation
- Process isolation adds overhead

## Sample Output
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

## Usage

1. Training with parallel processing:
```bash
python train.py --workers 8  # Adjust worker count based on CPU cores
```

2. Play against AI:
```bash
python chess_engine/main.py --mode ai
```

3. Watch AI self-play:
```bash
python chess_engine/main.py --mode auto
```

## Project Structure
- `chess_engine/`: Core chess engine implementation
  - `bitboard.py`: Efficient chess board representation
  - `game.py`: High-level chess game interface
  - `main.py`: CLI interface for playing/watching games
- `model/`: Neural network and learning components
  - `network.py`: MLX neural network architecture
  - `mcts.py`: Monte Carlo Tree Search implementation
  - `self_play.py`: Self-play game generation
  - `training.py`: Training loop implementation
- `config/`: Configuration files
- `utils/`: Utility functions
- `tests/`: Unit tests
- `train.py`: Main training script