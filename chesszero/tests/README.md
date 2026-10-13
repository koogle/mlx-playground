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
Rule tests run against our own `BitBoard` implementation. Install the independent,
test-only reference with `pip install -r tests/requirements.txt`, then run
`python -m unittest discover -s tests` from `chesszero/`. `python-chess` is never
imported by the engine, search, or training code.

Self-play automatically claims a reached threefold repetition or 100-halfmove
draw; there is no separate claim action in the policy. Checkmate takes precedence.
The input retains 19 planes; plane 18 now means halfmoves/100, and plane 17 no
longer saturates at 200 plies. Old checkpoints remain shape-compatible, but those
counter features have corrected semantics. En passant is tracked in board state
and legality/hash keys, but has no dedicated neural input plane.
