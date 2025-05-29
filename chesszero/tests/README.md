# Tests

This directory contains test suites for the ChessZero project.

## Test Files

### BitBoard Tests (`test_bitboard.py`)
- Tests for the BitBoard implementation
- Validates move generation
- Tests board state representation
- Verifies legal move detection
- Checks game state tracking (check, checkmate, draw)

### Board Tests (`test_board.py`)
- Tests for legacy board representation
- Provides compatibility testing

### Game Tests (`test_game.py`)
- Tests for the ChessGame class
- Validates game history tracking
- Tests algebraic notation conversion
- Verifies game state management
- Tests PGN export functionality

## Running Tests

To run all tests:
```bash
python -m pytest tests/
```

To run a specific test file:
```bash
python -m pytest tests/test_bitboard.py
```

To run tests with verbose output:
```bash
python -m pytest tests/ -v
```