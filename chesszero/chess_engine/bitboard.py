import numpy as np
from typing import List, Tuple, Set, Optional, Union
from dataclasses import dataclass
from functools import lru_cache


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
        # Use contiguous memory layout
        self.state = np.zeros((19, 8, 8), dtype=np.float32, order="C")
        # Add pre-computed attack tables
        self.knight_attacks = self._init_knight_attacks()
        self.king_attacks = self._init_king_attacks()
        self.pawn_attacks = self._init_pawn_attacks()

        # Cache structures
        self._moves_cache = {}
        self._game_over_cache = {}
        self._in_check_cache = {}
        self._valid_moves_count = None

        # Track king positions directly - (row, col) for each color
        self.king_positions = {
            0: (0, 4),  # White king starting position
            1: (7, 4),  # Black king starting position
        }

        self._pin_cache = {}  # Cache for pin calculations
        self._attack_cache = {}  # Cache for attack calculations

        self.initialize_board()

    def initialize_board(self):
        """Set up the initial chess position"""
        # White pieces (channels 0-5)
        white_pieces = self._init_pieces(0)
        black_pieces = self._init_pieces(1)

        self.state[0:6] = white_pieces
        self.state[6:12] = black_pieces

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

    def get_piece_at(
        self, row: int, col: int, promotion_piece: int = None
    ) -> Tuple[int, int]:
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

    def make_move(
        self,
        from_pos: Tuple[int, int],
        to_pos: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> bool:
        """Make a move on the board. Returns True if successful.
        For pawn promotion, to_pos should be (row, col, promotion_piece_type)"""
        # First validate current board state
        white_king_count = np.sum(self.state[5])
        black_king_count = np.sum(self.state[11])
        if white_king_count != 1 or black_king_count != 1:
            print("\nInvalid board state detected before move!")
            print(f"White kings: {white_king_count}, Black kings: {black_king_count}")
            print(f"Attempted move: {from_pos} -> {to_pos}")
            print("Current position:")
            print(self)
            return False

        from_row, from_col = from_pos
        color, piece_type = self.get_piece_at(from_row, from_col)

        if color == -1 or color != self.get_current_turn():
            return False

        # Handle pawn promotion
        is_promotion = False
        promotion_piece = None
        if isinstance(to_pos, tuple) and len(to_pos) == 3:
            to_row, to_col, promotion_piece = to_pos
            is_promotion = True
        else:
            to_row, to_col = to_pos

        # Check destination square
        target_color, target_piece = self.get_piece_at(to_row, to_col)

        # Prevent capturing kings
        if target_piece == 5:  # King is piece_type 5
            print(f"Illegal move: Cannot capture king at {to_pos}")
            return False

        # Correct capture detection - only enemy pieces count
        is_capture = (target_color != -1) and (target_color != color)
        resets_progress = piece_type == 0 or is_capture

        # Clear destination square
        self.state[:12, to_row, to_col] = 0

        # Move piece
        if is_promotion and piece_type == 0:  # Pawn promotion
            # Place promoted piece
            self.state[
                promotion_piece if color == 0 else promotion_piece + 6, to_row, to_col
            ] = 1
        else:
            # Move piece to target square
            self.state[piece_type if color == 0 else piece_type + 6, to_row, to_col] = 1

        # Clear source square
        self.state[piece_type if color == 0 else piece_type + 6, from_row, from_col] = 0

        # Validate board state after move
        white_king_count = np.sum(self.state[5])
        black_king_count = np.sum(self.state[11])
        if white_king_count != 1 or black_king_count != 1:
            print("\nInvalid board state after move!")
            print(f"White kings: {white_king_count}, Black kings: {black_king_count}")
            print(f"Move that caused error: {from_pos} -> {to_pos}")
            print("Resulting position:")
            print(self)
            # Revert the move
            self.state[piece_type if color == 0 else piece_type + 6, to_row, to_col] = 0
            self.state[
                piece_type if color == 0 else piece_type + 6, from_row, from_col
            ] = 1
            if is_capture:  # TODO: Need to track what piece was captured
                pass  # Would need to restore captured piece
            return False

        # Update king position if moving king
        if piece_type == 5:  # King
            self.king_positions[color] = (to_row, to_col)

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

        # Clear caches
        self._game_over_cache.clear()
        self._in_check_cache.clear()
        self._valid_moves_count = None

        # Clear caches after successful move
        self._pin_cache.clear()
        self._attack_cache.clear()
        self._moves_cache.clear()
        self._game_over_cache.clear()
        self._in_check_cache.clear()

        return True

    def get_move_count(self) -> int:
        """Get total number of moves played"""
        return int(self.state[17, 0, 0] * 200)  # Denormalize

    def get_moves_without_progress(self) -> int:
        """Get number of moves without a pawn move or capture"""
        return int(self.state[18, 0, 0] * 50)  # Denormalize

    def copy(self) -> "BitBoard":
        """Create a deep copy of the board"""
        new_board = BitBoard()
        new_board.state = np.copy(self.state)
        new_board.king_positions = self.king_positions.copy()
        # Reset caches for the copy
        new_board._moves_cache = {}
        new_board._game_over_cache = {}
        new_board._in_check_cache = {}
        return new_board

    @lru_cache(maxsize=1024)
    def get_valid_moves(self, pos: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Cache valid moves for positions"""
        row, col = pos
        color, piece_type = self.get_piece_at(row, col)

        if color == -1 or color != self.get_current_turn():
            return set()

        # Generate basic moves first
        moves = set()

        # Map piece types to pattern indices
        pattern_map = {
            0: 0,  # Pawn
            1: 1,  # Knight
            2: 2,  # Bishop
            3: 3,  # Rook
            4: 4,  # Queen
            5: 5,  # King
        }

        # Handle pawns separately due to special rules
        if piece_type == 0:  # Pawn
            moves = self._get_pawn_moves(row, col, color)
        else:
            pattern = PIECE_PATTERNS[pattern_map[piece_type]]
            if pattern.sliding:
                moves = self._get_sliding_moves(row, col, pattern.directions, color)
            else:
                moves = self._get_step_moves(row, col, pattern.directions, color)

        # Filter out moves that would capture the enemy king
        enemy_king_pos = self.king_positions[1 - color]
        if enemy_king_pos in moves:
            moves.remove(enemy_king_pos)

        # Check if piece is pinned
        pin_direction = self._get_pin_info(pos)
        if pin_direction:
            moves = self._filter_pinned_moves(pos, moves, pin_direction)

        # Filter moves that would leave king in check
        if self.is_in_check(color):
            moves = self._get_check_resolving_moves(pos, moves)

        return moves

    def _get_pawn_moves(self, row: int, col: int, color: int) -> Set[Tuple[int, int]]:
        """Get all valid pawn moves including captures and promotions"""
        moves = set()
        direction = (
            1 if color == 0 else -1
        )  # White moves up (+1), Black moves down (-1)
        start_row = 1 if color == 0 else 6
        promotion_row = 7 if color == 0 else 0

        # Forward moves
        new_row = row + direction
        if 0 <= new_row < 8:
            target_color = self.get_piece_at(new_row, col)[0]
            if target_color == -1:  # Empty square
                # Add promotion moves if pawn reaches last rank
                if new_row == promotion_row:
                    # Add all possible promotion moves (to Queen, Rook, Bishop, Knight)
                    for promotion_piece in range(
                        1, 5
                    ):  # 1=Knight, 2=Bishop, 3=Rook, 4=Queen
                        moves.add((new_row, col, promotion_piece))
                else:
                    moves.add((new_row, col))
                    # Double move from start
                    if row == start_row:
                        two_forward = row + 2 * direction
                        if 0 <= two_forward < 8:
                            two_forward_color = self.get_piece_at(two_forward, col)[0]
                            if two_forward_color == -1:
                                moves.add((two_forward, col))

        # Diagonal captures
        for col_delta in [-1, 1]:
            capture_col = col + col_delta
            if 0 <= capture_col < 8:
                capture_row = row + direction
                if 0 <= capture_row < 8:
                    target_color, _ = self.get_piece_at(capture_row, capture_col)
                    if target_color == (1 - color):  # Enemy piece
                        # Add promotion captures if pawn reaches last rank
                        if capture_row == promotion_row:
                            for promotion_piece in range(1, 5):
                                moves.add((capture_row, capture_col, promotion_piece))
                        else:
                            moves.add((capture_row, capture_col))

        return moves

    def _get_step_moves(
        self, row: int, col: int, directions: List[Tuple[int, int]], color: int
    ) -> Set[Tuple[int, int]]:
        """Get moves for non-sliding pieces (knight, king)"""
        moves = set()
        _, piece_type = self.get_piece_at(row, col)

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target_color, _ = self.get_piece_at(new_row, new_col)
                if target_color == -1 or target_color == (
                    1 - color
                ):  # Empty or enemy square
                    # For king moves, check if target square is under attack
                    if piece_type == 5:  # King
                        # Create a copy of the board and simulate the move
                        test_board = self.copy()
                        test_board.state[5 if color == 0 else 11, row, col] = (
                            0  # Remove king from current pos
                        )
                        test_board.state[5 if color == 0 else 11, new_row, new_col] = (
                            1  # Place king at new pos
                        )
                        test_board.king_positions[color] = (
                            new_row,
                            new_col,
                        )  # Update king position

                        # Only add move if target square is not under attack
                        if not test_board.is_square_attacked(
                            (new_row, new_col), 1 - color
                        ):
                            moves.add((new_row, new_col))
                    else:
                        moves.add((new_row, new_col))

        return moves

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

    def get_king_position(self, color: int) -> Optional[Tuple[int, int]]:
        """Return the position of the king for the given color.
        Returns None if the king is not found (error state)."""
        # For white, king is stored in channel 5; for black, in channel 11.
        channel = 5 if color == 0 else 11
        king_locations = np.argwhere(self.state[channel] == 1)
        if king_locations.size == 0:
            return None
        # Return as (row, col)
        return tuple(king_locations[0])

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

    @lru_cache(maxsize=2048)
    def is_square_attacked(self, pos: Tuple[int, int], by_color: int) -> bool:
        """Cache attack calculations"""
        row, col = pos

        # Check pawn attacks
        pawn_channel = 0 if by_color == 0 else 6
        pawn_direction = (
            -1 if by_color == 0 else 1
        )  # Pawns attack in opposite direction
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
                    )  # Look backwards
                    while 0 <= current_row < 8 and 0 <= current_col < 8:
                        if self.state[channel, current_row, current_col]:
                            return True
                        target_color, _ = self.get_piece_at(current_row, current_col)
                        if target_color != -1:  # Hit any piece
                            break
                        current_row -= row_delta
                        current_col -= col_delta
        return False

    def is_in_check(self, color: int) -> bool:
        """Check if the given color is in check"""
        king_pos = self.king_positions[color]
        return self.is_square_attacked(king_pos, 1 - color)

    def _filter_valid_moves(
        self, pos: Tuple[int, int], moves: Set[Tuple[int, int]]
    ) -> Set[Tuple[int, int]]:
        """Return only those moves which do not leave the king in check.
        It simulates each move on a copy of the board and uses is_in_check().
        """
        valid = set()
        current_turn = self.get_current_turn()
        for move in moves:
            new_board = self.copy()
            # Use make_move without printing or extra validation (it will update state)
            if new_board.make_move(pos, move):
                king_pos = new_board.get_king_position(current_turn)
                if king_pos is not None and not new_board.is_in_check(current_turn):
                    valid.add(move)
        return valid

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
        """Check if the given color is checkmated"""
        # First verify we're in check
        if not self.is_in_check(color):
            return False

        # Try all pieces' moves
        for pos, piece_type in self.get_all_pieces(color):
            moves = self.get_valid_moves(pos)
            if moves:
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
        """Check if the game is over (checkmate or draw)"""
        current_turn = self.get_current_turn()
        cache_key = hash(self.state.tobytes())

        if cache_key in self._game_over_cache:
            return self._game_over_cache[cache_key]

        # Check all pieces for valid moves
        has_moves = False
        for (r, c), piece_type in self.get_all_pieces(current_turn):
            if self.get_valid_moves((r, c)):
                has_moves = True
                break

        # Check for checkmate/stalemate
        result = not has_moves
        self._game_over_cache[cache_key] = result
        return result

    def _has_insufficient_material(self) -> bool:
        """Fast insufficient material check"""
        # Get piece counts (use numpy operations)
        white_pieces = self.state[0:6].sum()
        black_pieces = self.state[6:12].sum()

        # King vs King
        if white_pieces == 1 and black_pieces == 1:
            return True

        # If either side has more than 3 pieces, sufficient material
        if white_pieces > 3 or black_pieces > 3:
            return False

        # King + Bishop/Knight vs King
        if (white_pieces == 2 and black_pieces == 1) or (
            black_pieces == 2 and white_pieces == 1
        ):
            return True

        # King + Bishop vs King + Bishop (same color bishops)
        if white_pieces == 2 and black_pieces == 2:
            # Check if both extra pieces are bishops
            white_bishop = bool(self.state[2].any())  # Bishop channel
            black_bishop = bool(self.state[8].any())  # Bishop channel
            if white_bishop and black_bishop:
                # Check if bishops are on same colored squares
                white_bishop_pos = np.where(self.state[2])
                black_bishop_pos = np.where(self.state[8])
                if (white_bishop_pos[0] + white_bishop_pos[1]) % 2 == (
                    black_bishop_pos[0] + black_bishop_pos[1]
                ) % 2:
                    return True

        return False

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

    def get_hash(self) -> int:
        """Fast board hashing using numpy"""
        # Use numpy's built-in hashing for arrays
        return hash(self.state.tobytes())

    def _init_knight_attacks(self):
        """Pre-compute knight attack patterns for each square"""
        attacks = np.zeros((8, 8, 8, 8), dtype=np.bool_)  # from_square -> to_squares
        for row in range(8):
            for col in range(8):
                for dr, dc in PIECE_PATTERNS[1].directions:  # Knight patterns
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < 8 and 0 <= new_col < 8:
                        attacks[row, col, new_row, new_col] = True
        return attacks

    def _init_king_attacks(self):
        """Pre-compute king attack patterns for each square"""
        attacks = np.zeros((8, 8, 8, 8), dtype=np.bool_)  # from_square -> to_squares
        for row in range(8):
            for col in range(8):
                for dr, dc in PIECE_PATTERNS[5].directions:  # King patterns
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < 8 and 0 <= new_col < 8:
                        attacks[row, col, new_row, new_col] = True
        return attacks

    def _init_pawn_attacks(self):
        """Pre-compute pawn attack patterns for each square"""
        attacks = np.zeros((8, 8, 8, 8), dtype=np.bool_)  # from_square -> to_squares
        for row in range(8):
            for col in range(8):
                for dr, dc in PIECE_PATTERNS[0].directions:  # Pawn patterns
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < 8 and 0 <= new_col < 8:
                        attacks[row, col, new_row, new_col] = True
        return attacks

    def _make_move_on_board(
        self, board: np.ndarray, from_pos: Tuple[int, int], to_pos: Tuple[int, int]
    ):
        """Make move on a board array without validation"""
        from_row, from_col = from_pos
        to_row, to_col = to_pos

        # Get piece details
        color = -1
        piece_type = -1
        for pt in range(6):
            if board[pt, from_row, from_col]:
                color = 0
                piece_type = pt
                break
            elif board[pt + 6, from_row, from_col]:
                color = 1
                piece_type = pt
                break

        if color == -1:
            return

        # Clear source square
        channel = piece_type if color == 0 else piece_type + 6
        board[channel, from_row, from_col] = 0

        # Clear destination square (capture)
        board[:12, to_row, to_col] = 0

        # Place piece at destination
        board[channel, to_row, to_col] = 1

    def _is_in_check_vectorized(self, boards: np.ndarray, color: int) -> np.ndarray:
        """Check if positions are in check (vectorized)"""
        batch_size = boards.shape[0]
        in_check = np.zeros(batch_size, dtype=bool)

        # Find kings
        king_channel = 5 if color == 0 else 11
        for i in range(batch_size):
            king_pos = np.where(boards[i, king_channel] > 0)
            if len(king_pos[0]) == 0:
                in_check[i] = True
                continue
            king_row, king_col = king_pos[0][0], king_pos[1][0]

            # Check if king is attacked
            enemy_color = 1 - color
            in_check[i] = self._is_square_attacked_vectorized(
                boards[i], (king_row, king_col), enemy_color
            )

        return in_check

    def _is_square_attacked_vectorized(
        self, pos: Tuple[int, int], by_color: int
    ) -> bool:
        """Vectorized attack detection using numpy operations"""
        row, col = pos

        # Create attack masks for each piece type
        pawn_mask = self._get_pawn_attack_mask(row, col, by_color)
        knight_mask = self._get_knight_attack_mask(row, col)
        bishop_mask = self._get_bishop_attack_mask(row, col)
        rook_mask = self._get_rook_attack_mask(row, col)
        queen_mask = bishop_mask | rook_mask
        king_mask = self._get_king_attack_mask(row, col)

        # Get piece positions for attacking color
        piece_positions = self.state[6 * by_color : 6 * (by_color + 1)]

        # Check attacks using vectorized operations
        return (
            np.any(pawn_mask & piece_positions[0])
            or np.any(knight_mask & piece_positions[1])
            or np.any(bishop_mask & piece_positions[2])
            or np.any(rook_mask & piece_positions[3])
            or np.any(queen_mask & piece_positions[4])
            or np.any(king_mask & piece_positions[5])
        )

    def get_game_result(self) -> float:
        """Get the game result from current player's perspective
        Returns:
            1.0 for win
            -1.0 for loss
            0.0 for draw
        """
        current_turn = self.get_current_turn()
        opponent_turn = 1 - current_turn

        # Check if current player is checkmated
        if self.is_checkmate(current_turn):
            return -1.0  # Current player is checkmated

        # Check if opponent is checkmated
        elif self.is_checkmate(opponent_turn):
            return 1.0  # Current player delivered checkmate

        # Check for draw
        elif self.is_draw():
            return 0.0  # Draw

        # If game is not over, return 0.0
        return 0.0

    def _get_ray_between(
        self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]
    ) -> Set[Tuple[int, int]]:
        """Get all squares between two positions (exclusive) along a straight line"""
        from_row, from_col = from_pos
        to_row, to_col = to_pos

        # Check if positions are aligned
        if from_row == to_row:
            step = 1 if to_col > from_col else -1
            return {(from_row, col) for col in range(from_col + step, to_col, step)}
        elif from_col == to_col:
            step = 1 if to_row > from_row else -1
            return {(row, from_col) for row in range(from_row + step, to_row, step)}
        elif abs(to_row - from_row) == abs(to_col - from_col):
            row_step = 1 if to_row > from_row else -1
            col_step = 1 if to_col > from_col else -1
            return {
                (from_row + i * row_step, from_col + i * col_step)
                for i in range(1, abs(to_row - from_row))
            }
        return set()  # Not aligned

    def _is_pinned(self, pos: Tuple[int, int]) -> bool:
        """Check if a piece is pinned to the king"""
        row, col = pos
        color, piece_type = self.get_piece_at(row, col)
        if color == -1 or piece_type == 5:  # King can't be pinned
            return False

        king_pos = self.king_positions[color]
        attacker_color = 1 - color

        # Check if there's a line from an attacker to the king through this piece
        attackers = self.get_all_pieces(attacker_color)
        for (ar, ac), atype in attackers:
            ray = self._get_ray_between((ar, ac), king_pos)
            if ray and (row, col) in ray:
                return True
        return False

    def _get_check_resolving_moves(
        self, pos: Tuple[int, int], moves: Set[Tuple[int, int]]
    ) -> Set[Tuple[int, int]]:
        """Get moves that help resolve a check situation"""
        color = self.get_current_turn()
        valid_moves = set()

        # Try each move and see if it actually resolves the check
        for move in moves:
            # Create a copy of the board and try the move
            test_board = self.copy()
            if test_board.make_move(pos, move):
                if not test_board.is_in_check(color):
                    valid_moves.add(move)

        return valid_moves

    def _get_pin_info(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Cache pin information"""
        board_hash = self.get_hash()
        cache_key = (board_hash, pos)

        if cache_key in self._pin_cache:
            return self._pin_cache[cache_key]

        result = self._calculate_pin_info(pos)
        self._pin_cache[cache_key] = result
        return result

    def _calculate_pin_info(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Get pin direction if piece is pinned, returns (dr, dc) or None"""
        row, col = pos
        color, piece_type = self.get_piece_at(row, col)
        if color == -1 or piece_type == 5:  # King can't be pinned
            return None

        king_pos = self.king_positions[color]
        k_row, k_col = king_pos
        attacker_color = 1 - color

        # Check all potential attackers
        for (ar, ac), piece_type in self.get_all_pieces(attacker_color):
            ray = self._get_ray_between((ar, ac), king_pos)
            if (row, col) in ray:
                # Calculate pin direction (king to attacker vector)
                dr = ar - k_row
                dc = ac - k_col
                if dr != 0:
                    dr //= abs(dr)
                if dc != 0:
                    dc //= abs(dc)
                return (dr, dc)
        return None

    def _filter_pinned_moves(
        self,
        pos: Tuple[int, int],
        moves: Set[Tuple[int, int] | Tuple[int, int, int]],
        direction: Tuple[int, int],
    ) -> Set[Tuple[int, int]]:
        """Filter moves for a pinned piece.
        Only allows moves along the pin line between king and attacker."""
        row, col = pos
        king_pos = self.king_positions[self.get_current_turn()]
        kr, kc = king_pos
        dr, dc = direction

        valid = set()
        for move in moves:
            if len(move) == 2:
                mr, mc = move
            else:
                mr, mc, _ = move

            # Move must be along the same line as the pin
            if dr != 0:  # Vertical pin
                if mc == col:  # Must stay in same column
                    # Must stay between king and attacker
                    min_row = min(kr, row + dr * 8)  # Attacker's row
                    max_row = max(kr, row + dr * 8)
                    if min_row <= mr <= max_row:
                        valid.add(move)
            elif dc != 0:  # Horizontal pin
                if mr == row:  # Must stay in same row
                    min_col = min(kc, col + dc * 8)  # Attacker's col
                    max_col = max(kc, col + dc * 8)
                    if min_col <= mc <= max_col:
                        valid.add(move)

        return valid

    def _calculate_pawn_attacks(self, row: int, col: int) -> int:
        """Calculate pawn attacks for both colors"""
        attacks = 0
        # White pawns (attack up-left/up-right)
        if row < 7:
            if col > 0:
                attacks |= 1 << ((row + 1) * 8 + (col - 1))  # Up-left
            if col < 7:
                attacks |= 1 << ((row + 1) * 8 + (col + 1))  # Up-right
        # Black pawns (attack down-left/down-right)
        if row > 0:
            if col > 0:
                attacks |= 1 << ((row - 1) * 8 + (col - 1))  # Down-left
            if col < 7:
                attacks |= 1 << ((row - 1) * 8 + (col + 1))  # Down-right
        return attacks
