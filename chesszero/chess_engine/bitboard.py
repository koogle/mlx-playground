import numpy as np
from enum import Enum
from typing import List, Tuple, Set
from dataclasses import dataclass


@dataclass
class MovePattern:
    """Represents a movement pattern for a piece"""

    directions: List[Tuple[int, int]]  # (row_delta, col_delta)
    sliding: bool  # Whether piece can move multiple squares in these directions


PIECE_PATTERNS = {
    # Piece type index -> movement pattern
    0: MovePattern([(1, 0)], False),  # White pawns move up
    1: MovePattern(
        [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)], False
    ),  # Knights
    2: MovePattern([(1, 1), (1, -1), (-1, 1), (-1, -1)], True),  # Bishops
    3: MovePattern([(1, 0), (-1, 0), (0, 1), (0, -1)], True),  # Rooks
    4: MovePattern(
        [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)], True
    ),  # Queens
    5: MovePattern(
        [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)], False
    ),  # Kings
}


class BitBoard:
    """
    Chess board representation using bit boards for efficient move generation and state updates.
    Uses 14 channels:
    0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    6-11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    12: Current turn (1 for white, 0 for black)
    13: Castle rights
    """

    def __init__(self):
        # Initialize board state as numpy array for efficient operations
        self.state = np.zeros((14, 8, 8), dtype=np.uint8)
        self.initialize_board()

    def initialize_board(self):
        """Set up the initial chess position"""
        # White pieces (channels 0-5)
        # Pawns
        self.state[0, 1, :] = 1
        # Knights
        self.state[1, 0, [1, 6]] = 1
        # Bishops
        self.state[2, 0, [2, 5]] = 1
        # Rooks
        self.state[3, 0, [0, 7]] = 1
        # Queen
        self.state[4, 0, 3] = 1
        # King
        self.state[5, 0, 4] = 1

        # Black pieces (channels 6-11)
        # Pawns
        self.state[6, 6, :] = 1
        # Knights
        self.state[7, 7, [1, 6]] = 1
        # Bishops
        self.state[8, 7, [2, 5]] = 1
        # Rooks
        self.state[9, 7, [0, 7]] = 1
        # Queen
        self.state[10, 7, 3] = 1
        # King
        self.state[11, 7, 4] = 1

        # White to move
        self.state[12] = 1

        # Initial castling rights - set for king and rook squares
        self.state[13] = 0  # Clear all castling rights first
        self.state[13, 0, [0, 4, 7]] = 1  # White castling rights
        self.state[13, 7, [0, 4, 7]] = 1  # Black castling rights

    def get_piece_at(self, row: int, col: int) -> Tuple[int, int]:
        """Returns (color, piece_type) at given position, where:
        color: 0 for white, 1 for black
        piece_type: 0-5 for pawn through king
        Returns (-1, -1) if square is empty
        """
        # Check white pieces
        for piece_type in range(6):
            if self.state[piece_type, row, col]:
                return (0, piece_type)

        # Check black pieces
        for piece_type in range(6):
            if self.state[piece_type + 6, row, col]:
                return (1, piece_type)

        return (-1, -1)

    def get_current_turn(self) -> int:
        """Returns 0 for white, 1 for black"""
        return 0 if self.state[12, 0, 0] else 1

    def make_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """Make a move on the board. Returns True if successful."""
        from_row, from_col = from_pos
        to_row, to_col = to_pos

        # Get piece details
        color, piece_type = self.get_piece_at(from_row, from_col)
        if color == -1 or color != self.get_current_turn():
            return False

        # Clear source square
        channel = piece_type if color == 0 else piece_type + 6
        self.state[channel, from_row, from_col] = 0

        # Clear destination square (capture)
        for c in range(12):
            self.state[c, to_row, to_col] = 0

        # Place piece at destination
        self.state[channel, to_row, to_col] = 1

        # Update castling rights
        if piece_type == 5:  # King move
            row = 0 if color == 0 else 7
            self.state[13, row, :] = 0
        elif piece_type == 3:  # Rook move
            row = 0 if color == 0 else 7
            if from_col == 0:  # Queenside rook
                self.state[13, row, [0, 4]] = 0
            elif from_col == 7:  # Kingside rook
                self.state[13, row, [4, 7]] = 0

        # Update turn
        self.state[12] = 1 - self.state[12]

        return True

    def copy(self) -> "BitBoard":
        """Create a deep copy of the board"""
        new_board = BitBoard.__new__(BitBoard)
        new_board.state = self.state.copy()
        return new_board

    def get_valid_moves(self, pos: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Get all valid moves for a piece at the given position, considering check"""
        row, col = pos
        color, piece_type = self.get_piece_at(row, col)
        if color == -1 or color != self.get_current_turn():
            return set()

        # Get basic moves first
        moves = set()

        # Handle pawns separately due to special rules
        if piece_type == 0:  # Pawn
            moves.update(self._get_pawn_moves(row, col, color))
        else:
            pattern = PIECE_PATTERNS[piece_type]
            # Generate moves based on piece pattern
            if pattern.sliding:
                moves.update(
                    self._get_sliding_moves(row, col, pattern.directions, color)
                )
            else:
                moves.update(self._get_step_moves(row, col, pattern.directions, color))

        # Add castling moves for king
        if piece_type == 5 and not self.is_in_check(color):  # King, not in check
            row = 0 if color == 0 else 7
            if self.can_castle_kingside(color):
                moves.add((row, 6))
            if self.can_castle_queenside(color):
                moves.add((row, 2))

        # Filter moves that would leave king in check
        return self._filter_valid_moves(pos, moves)

    def _get_sliding_moves(
        self, row: int, col: int, directions: List[Tuple[int, int]], color: int
    ) -> Set[Tuple[int, int]]:
        """Get moves for sliding pieces (bishop, rook, queen)"""
        moves = set()

        for row_delta, col_delta in directions:
            current_row, current_col = row + row_delta, col + col_delta

            while 0 <= current_row < 8 and 0 <= current_col < 8:
                target_color, _ = self.get_piece_at(current_row, current_col)

                if target_color == -1:  # Empty square
                    moves.add((current_row, current_col))
                elif target_color == color:  # Own piece
                    break
                else:  # Enemy piece
                    moves.add((current_row, current_col))
                    break

                current_row += row_delta
                current_col += col_delta

        return moves

    def _get_step_moves(
        self, row: int, col: int, directions: List[Tuple[int, int]], color: int
    ) -> Set[Tuple[int, int]]:
        """Get moves for non-sliding pieces (king, knight)"""
        moves = set()

        for row_delta, col_delta in directions:
            new_row, new_col = row + row_delta, col + col_delta

            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target_color, _ = self.get_piece_at(new_row, new_col)
                if target_color != color:  # Empty or enemy piece
                    moves.add((new_row, new_col))

        return moves

    def _get_pawn_moves(self, row: int, col: int, color: int) -> Set[Tuple[int, int]]:
        """Get valid pawn moves including captures and initial double move"""
        moves = set()
        direction = 1 if color == 0 else -1  # White moves up, black moves down
        start_row = 1 if color == 0 else 6

        # Forward move
        new_row = row + direction
        if 0 <= new_row < 8:
            # Check if square in front is empty
            if self.get_piece_at(new_row, col)[0] == -1:  # Square must be empty
                moves.add((new_row, col))

                # Initial two-square move - only if path is clear and on starting rank
                if row == start_row:
                    two_forward = row + 2 * direction
                    # Check if both squares are empty
                    if (
                        self.get_piece_at(new_row, col)[0] == -1
                        and self.get_piece_at(two_forward, col)[0] == -1
                    ):
                        moves.add((two_forward, col))

        # Captures
        for col_delta in [-1, 1]:
            new_col = col + col_delta
            if 0 <= new_col < 8 and 0 <= new_row < 8:
                target_color, _ = self.get_piece_at(new_row, new_col)
                if target_color == (1 - color):  # Enemy piece
                    moves.add((new_row, new_col))

        return moves

    def get_king_position(self, color: int) -> Tuple[int, int]:
        """Find the position of the king for the given color"""
        channel = 5 if color == 0 else 11
        king_pos = np.where(self.state[channel] == 1)
        if len(king_pos[0]) == 0:
            raise ValueError(f"No king found for color {color}")
        return (int(king_pos[0][0]), int(king_pos[1][0]))

    def get_all_pieces(self, color: int) -> List[Tuple[Tuple[int, int], int]]:
        """Get all pieces for the given color as [(position, piece_type),...]"""
        pieces = []
        start_channel = 0 if color == 0 else 6

        for piece_type in range(6):
            channel = start_channel + piece_type
            positions = np.where(self.state[channel] == 1)
            for i in range(len(positions[0])):
                pos = (int(positions[0][i]), int(positions[1][i]))
                pieces.append((pos, piece_type))

        return pieces

    def is_square_attacked(self, pos: Tuple[int, int], by_color: int) -> bool:
        """Check if a square is attacked by any piece of the given color"""
        row, col = pos

        # Check pawn attacks
        pawn_channel = 0 if by_color == 0 else 6
        pawn_direction = (
            -1 if by_color == 0 else 1
        )  # CHANGED: Pawns attack in opposite direction
        for col_delta in [-1, 1]:
            attack_row = row + pawn_direction
            attack_col = col + col_delta
            if 0 <= attack_row < 8 and 0 <= attack_col < 8:
                if self.state[pawn_channel, attack_row, attack_col]:
                    return True

        # Check other piece attacks
        for piece_type in range(1, 6):  # Skip pawns, already checked
            channel = piece_type if by_color == 0 else piece_type + 6
            pattern = PIECE_PATTERNS[piece_type]

            if pattern.sliding:
                # For sliding pieces, check along each direction until we hit something
                for row_delta, col_delta in pattern.directions:
                    current_row, current_col = (
                        row - row_delta,
                        col - col_delta,
                    )  # CHANGED: Look backwards
                    while 0 <= current_row < 8 and 0 <= current_col < 8:
                        if self.state[channel, current_row, current_col]:
                            return True
                        target_color, _ = self.get_piece_at(current_row, current_col)
                        if target_color != -1:  # Hit any piece
                            break
                        current_row -= row_delta  # CHANGED: Move backwards
                        current_col -= col_delta
            else:
                # For non-sliding pieces, just check the pattern squares
                for row_delta, col_delta in pattern.directions:
                    attack_row = row + row_delta
                    attack_col = col + col_delta
                    if 0 <= attack_row < 8 and 0 <= attack_col < 8:
                        if self.state[channel, attack_row, attack_col]:
                            return True

        return False

    def is_in_check(self, color: int) -> bool:
        """Check if the given color's king is in check"""
        king_pos = self.get_king_position(color)
        return self.is_square_attacked(king_pos, 1 - color)

    def _filter_valid_moves(
        self, pos: Tuple[int, int], moves: Set[Tuple[int, int]]
    ) -> Set[Tuple[int, int]]:
        """Filter moves that would leave the king in check"""
        color, _ = self.get_piece_at(*pos)
        valid_moves = set()

        for move in moves:
            # Try the move
            board_copy = self.copy()
            board_copy.make_move(pos, move)

            # If it doesn't leave us in check, it's valid
            if not board_copy.is_in_check(color):
                valid_moves.add(move)

        return valid_moves

    def can_castle_kingside(self, color: int) -> bool:
        """Check if kingside castling is possible"""
        row = 0 if color == 0 else 7
        king_channel = 5 if color == 0 else 11
        rook_channel = 3 if color == 0 else 9

        # Check pieces and rights
        if not self.state[king_channel, row, 4]:
            return False
        if not self.state[rook_channel, row, 7]:
            return False
        if not (self.state[13, row, 4] and self.state[13, row, 7]):
            return False

        # Check squares between are empty
        if any(self.get_piece_at(row, col)[0] != -1 for col in range(5, 7)):
            return False

        # Check path not under attack
        enemy_color = 1 - color
        return not any(
            self.is_square_attacked((row, col), enemy_color) for col in range(4, 7)
        )

    def can_castle_queenside(self, color: int) -> bool:
        """Check if queenside castling is possible"""
        row = 0 if color == 0 else 7
        king_channel = 5 if color == 0 else 11
        rook_channel = 3 if color == 0 else 9

        # Check if king and rook are in original positions
        if not (self.state[king_channel, row, 4] and self.state[rook_channel, row, 0]):
            return False

        # Check if castling rights are maintained
        if not (self.state[13, row, 4] and self.state[13, row, 0]):
            return False

        # Check if squares between are empty
        if any(self.get_piece_at(row, col)[0] != -1 for col in range(1, 4)):
            return False

        # Check if king's path is under attack
        enemy_color = 1 - color
        return not any(
            self.is_square_attacked((row, col), enemy_color) for col in range(2, 5)
        )

    def is_checkmate(self, color: int) -> bool:
        """Check if the given color is in checkmate"""
        # First verify the king is in check
        if not self.is_in_check(color):
            return False

        # Check if any piece has valid moves
        for piece_pos, _ in self.get_all_pieces(color):
            if self.get_valid_moves(piece_pos):
                return False

        return True

    def is_stalemate(self, color: int) -> bool:
        """Check if the given color is in stalemate"""
        # If in check, it's not stalemate
        if self.is_in_check(color):
            return False

        # Check if any piece has valid moves
        for piece_pos, _ in self.get_all_pieces(color):
            if self.get_valid_moves(piece_pos):
                return False

        return True

    def test_blocked_pawn(self):
        """Test blocked pawn movement"""
        # Block a white pawn with a black pawn
        self.state[6, 2, 0] = 1  # Put a black pawn in front of a2 pawn
        moves = self.get_valid_moves((1, 0))
        assert len(moves) == 0  # Pawn should have no valid moves

    def test_check_detection(self):
        """Test check detection with queen attacking king"""
        # Clear the board first
        self.state.fill(0)

        # Set up a simple position:
        # White queen at e4 attacking black king at e8
        self.state[4, 3, 4] = 1  # White queen at e4
        self.state[11, 7, 4] = 1  # Black king at e8
        self.state[5, 0, 4] = 1  # White king at e1 (needed!)
        self.state[12] = 1  # White to move
