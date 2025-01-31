import numpy as np
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
    Uses 19 channels:
    0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    6-11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    12: Color to move (1 for white, 0 for black)
    13-16: Castling rights (WK, WQ, BK, BQ)
    17: Move count (normalized by dividing by max moves)
    18: No-progress count (normalized by dividing by 50)
    """

    def __init__(self):
        # Initialize board state as numpy array for efficient operations
        self.state = np.zeros((19, 8, 8), dtype=np.float32)  # Changed to float32
        self.initialize_board()

    def initialize_board(self):
        """Set up the initial chess position"""
        # White pieces (channels 0-5)
        self.state[0:6] = self._init_pieces(0)  # White pieces
        self.state[6:12] = self._init_pieces(1)  # Black pieces

        # White to move
        self.state[12].fill(1)  # White to move = 1

        # Initialize castling rights (one plane each)
        self.state[13].fill(1)  # White kingside
        self.state[14].fill(1)  # White queenside
        self.state[15].fill(1)  # Black kingside
        self.state[16].fill(1)  # Black queenside

        # Move counters start at 0
        self.state[17:].fill(0)

    def _init_pieces(self, color: int) -> np.ndarray:
        """Initialize piece positions for given color"""
        pieces = np.zeros((6, 8, 8), dtype=np.float32)
        rank = 0 if color == 0 else 7
        pawn_rank = 1 if color == 0 else 6

        # Pawns
        pieces[0, pawn_rank] = 1
        # Knights
        pieces[1, rank, [1, 6]] = 1
        # Bishops
        pieces[2, rank, [2, 5]] = 1
        # Rooks
        pieces[3, rank, [0, 7]] = 1
        # Queen
        pieces[4, rank, 3] = 1
        # King
        pieces[5, rank, 4] = 1

        return pieces

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

        # Check if move resets progress counter (pawn move or capture)
        is_capture = any(self.state[c, to_row, to_col] == 1 for c in range(12))
        resets_progress = piece_type == 0 or is_capture  # Pawn move or capture

        # Clear source square
        channel = piece_type if color == 0 else piece_type + 6
        self.state[channel, from_row, from_col] = 0

        # Clear destination square (capture)
        for c in range(12):
            self.state[c, to_row, to_col] = 0

        # Place piece at destination
        self.state[channel, to_row, to_col] = 1

        # Update castling rights (now using separate planes)
        if piece_type == 5:  # King move
            if color == 0:
                self.state[13:15].fill(0)  # Clear white castling
            else:
                self.state[15:17].fill(0)  # Clear black castling
        elif piece_type == 3:  # Rook move
            if color == 0:
                if from_col == 0:
                    self.state[14].fill(0)  # Clear white queenside
                elif from_col == 7:
                    self.state[13].fill(0)  # Clear white kingside
            else:
                if from_col == 0:
                    self.state[16].fill(0)  # Clear black queenside
                elif from_col == 7:
                    self.state[15].fill(0)  # Clear black kingside

        # Update move counters (normalized)
        moves = self.get_move_count() + 1
        self.state[17].fill(min(moves / 200, 1.0))  # Normalize by max moves

        if resets_progress:
            self.state[18].fill(0)
        else:
            progress = self.get_moves_without_progress() + 1
            self.state[18].fill(min(progress / 50, 1.0))  # Normalize by 50 move rule

        # Update turn
        self.state[12] = 1 - self.state[12]

        return True

    def get_move_count(self) -> int:
        """Get total number of moves played"""
        return int(self.state[17, 0, 0] * 200)  # Denormalize

    def get_moves_without_progress(self) -> int:
        """Get number of moves without a pawn move or capture"""
        return int(self.state[18, 0, 0] * 50)  # Denormalize

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

                # Initial two-square move - only if pawn is on its starting rank
                if row == start_row:  # Changed condition to check starting rank
                    two_forward = row + 2 * direction
                    # Check if both squares are empty
                    if (
                        0 <= two_forward < 8  # Make sure we're on the board
                        and self.get_piece_at(new_row, col)[0]
                        == -1  # First square empty
                        and self.get_piece_at(two_forward, col)[0]
                        == -1  # Second square empty
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

    def is_draw(self) -> bool:
        """Check if the position is a draw (stalemate or insufficient material)"""
        current_turn = self.get_current_turn()

        # Check for stalemate
        if self.is_stalemate(current_turn):
            return True

        # Get all pieces
        white_pieces = self.get_all_pieces(0)
        black_pieces = self.get_all_pieces(1)

        # Count pieces by type
        white_counts = {piece_type: 0 for piece_type in range(6)}
        black_counts = {piece_type: 0 for piece_type in range(6)}

        for _, piece_type in white_pieces:
            white_counts[piece_type] += 1
        for _, piece_type in black_pieces:
            black_counts[piece_type] += 1

        # Insufficient material cases:

        # 1. King vs King
        if len(white_pieces) == 1 and len(black_pieces) == 1:
            return True

        # 2. King and minor piece vs King
        if (len(white_pieces) == 2 and len(black_pieces) == 1) or (
            len(white_pieces) == 1 and len(black_pieces) == 2
        ):
            # Check if extra piece is a knight or bishop
            if (white_counts[1] == 1 or white_counts[2] == 1) or (
                black_counts[1] == 1 or black_counts[2] == 1
            ):
                return True

        # 3. King and bishop vs King and bishop (same colored squares)
        if len(white_pieces) == 2 and len(black_pieces) == 2:
            if white_counts[2] == 1 and black_counts[2] == 1:  # Both have bishops
                # Get bishop positions
                white_bishop_pos = next(
                    pos for pos, piece_type in white_pieces if piece_type == 2
                )
                black_bishop_pos = next(
                    pos for pos, piece_type in black_pieces if piece_type == 2
                )
                # Check if bishops are on same colored squares
                if (white_bishop_pos[0] + white_bishop_pos[1]) % 2 == (
                    black_bishop_pos[0] + black_bishop_pos[1]
                ) % 2:
                    return True

        # 4. King and knight vs King and knight
        if len(white_pieces) == 2 and len(black_pieces) == 2:
            if white_counts[1] == 1 and black_counts[1] == 1:  # Both have knights
                return True

        return False

    def is_game_over(self) -> bool:
        """Check if the game is over (checkmate, stalemate, draw, or max moves reached)"""
        current_turn = self.get_current_turn()
        return (
            self.is_checkmate(current_turn)
            or self.is_stalemate(current_turn)
            or self.is_draw()
            or self.get_moves_without_progress() >= 75  # 75-move rule
            or self.get_move_count() >= 200  # Maximum game length
        )

    def __str__(self) -> str:
        """Return string representation of the board"""
        # Unicode chess pieces
        piece_symbols = {
            (0, 0): "♙",  # White pawn
            (0, 1): "♘",  # White knight
            (0, 2): "♗",  # White bishop
            (0, 3): "♖",  # White rook
            (0, 4): "♕",  # White queen
            (0, 5): "♔",  # White king
            (1, 0): "♟",  # Black pawn
            (1, 1): "♞",  # Black knight
            (1, 2): "♝",  # Black bishop
            (1, 3): "♜",  # Black rook
            (1, 4): "♛",  # Black queen
            (1, 5): "♚",  # Black king
        }

        # Build the board string
        board_str = "\n  a b c d e f g h\n"
        for row in range(7, -1, -1):  # Start from rank 8
            board_str += f"{row + 1} "
            for col in range(8):
                piece = self.get_piece_at(row, col)
                if piece[0] == -1:  # Empty square
                    board_str += "· "
                else:
                    board_str += piece_symbols[piece] + " "
            board_str += f"{row + 1}\n"
        board_str += "  a b c d e f g h\n"

        # Add current turn
        turn = "White" if self.get_current_turn() == 0 else "Black"
        board_str += f"\n{turn} to move"

        return board_str
