# Chess Engine

This directory contains the core chess engine implementation for the ChessZero project.

## Components

### BitBoard (`bitboard.py`)
- Efficient chess board representation using bitboards
- 19-channel representation for the board state:
  - 12 channels for piece types/colors
  - Additional channels for game state (castling rights, etc.)
- Optimized move generation and validation
- Efficient board state management

### ChessGame (`game.py`)
- High-level interface for chess games
- Move representation in algebraic notation
- Game history tracking
- Draw detection
- PGN export support
- Uses BitBoard as the underlying representation

### Main CLI (`main.py`)
- Command-line interface for playing chess
- Supports multiple modes:
  - Human vs. Human
  - Human vs. AI
  - AI vs. AI (auto mode)
- Real-time board visualization
- Move history display

## Usage

### Play against AI:
```bash
python main.py --mode ai
```

### Watch AI self-play:
```bash
python main.py --mode auto
```

### Play human vs. human:
```bash
python main.py --mode human
```