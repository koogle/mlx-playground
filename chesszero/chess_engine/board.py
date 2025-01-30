from enum import Enum
from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass


class PieceType(Enum):
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6

    def get_move_patterns(self) -> List[Tuple[int, int]]:
        """Returns the basic movement patterns for each piece type."""
        patterns = {
            PieceType.PAWN: [],  # Special handling for pawns
            PieceType.KNIGHT: [
                (2, 1),
                (2, -1),
                (-2, 1),
                (-2, -1),
                (1, 2),
                (1, -2),
                (-1, 2),
                (-1, -2),
            ],
            PieceType.BISHOP: [(1, 1), (1, -1), (-1, 1), (-1, -1)],
            PieceType.ROOK: [(1, 0), (-1, 0), (0, 1), (0, -1)],
            PieceType.KING: [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ],
            PieceType.QUEEN: [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ],
        }
        return patterns[self]

    def is_sliding_piece(self) -> bool:
        """Returns True if the piece can move multiple squares in its direction."""
        return self in {PieceType.BISHOP, PieceType.ROOK, PieceType.QUEEN}


class Color(Enum):
    WHITE = 0
    BLACK = 1


class Piece:
    def __init__(self, piece_type: PieceType, color: Color):
        self.piece_type = piece_type
        self.color = color
        self.has_moved = False

    def __str__(self) -> str:
        return f"{self.color.name} {self.piece_type.name}"


@dataclass
class AttackInfo:
    """Information about attacks on the board"""

    attacked_squares: Set[Tuple[int, int]]  # All squares attacked by enemy pieces
    attacking_pieces: List[Tuple[Piece, Tuple[int, int]]]  # Pieces attacking the king
    pin_lines: Dict[
        Tuple[int, int], Set[Tuple[int, int]]
    ]  # Squares a pinned piece can move to


class Board:
    def __init__(self):
        self.squares = [[None for _ in range(8)] for _ in range(8)]
        self.white_pieces = []
        self.black_pieces = []
        self.current_turn = Color.WHITE
        self.initialize_board()

    def initialize_board(self):
        """Initialize the board with pieces in starting positions."""
        # Initialize pawns
        for col in range(8):
            white_pawn = Piece(PieceType.PAWN, Color.WHITE)
            black_pawn = Piece(PieceType.PAWN, Color.BLACK)
            self.squares[1][col] = white_pawn
            self.squares[6][col] = black_pawn
            self.white_pieces.append((white_pawn, (1, col)))
            self.black_pieces.append((black_pawn, (6, col)))

        # Initialize other pieces
        piece_order = [
            PieceType.ROOK,
            PieceType.KNIGHT,
            PieceType.BISHOP,
            PieceType.QUEEN,
            PieceType.KING,
            PieceType.BISHOP,
            PieceType.KNIGHT,
            PieceType.ROOK,
        ]

        for col in range(8):
            white_piece = Piece(piece_order[col], Color.WHITE)
            black_piece = Piece(piece_order[col], Color.BLACK)
            self.squares[0][col] = white_piece
            self.squares[7][col] = black_piece
            self.white_pieces.append((white_piece, (0, col)))
            self.black_pieces.append((black_piece, (7, col)))

    def get_valid_moves(self, pos: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Get all legal moves for a piece considering check and pins."""
        piece = self.squares[pos[0]][pos[1]]
        if not piece:
            return set()

        # Get basic moves and attack info
        basic_moves = self.get_basic_moves(pos)
        attack_info = self.get_attack_info(piece.color)

        # If piece is pinned, it can only move along the pin line
        if pos in attack_info.pin_lines:
            return basic_moves & attack_info.pin_lines[pos]

        # For king, filter out moves to attacked squares and add castling
        if piece.piece_type == PieceType.KING:
            valid_moves = {
                move for move in basic_moves if move not in attack_info.attacked_squares
            }

            # Add castling moves if possible
            if not piece.has_moved and not attack_info.attacking_pieces:
                if self._can_castle_kingside(pos, piece.color):
                    valid_moves.add((pos[0], pos[1] + 2))
                if self._can_castle_queenside(pos, piece.color):
                    valid_moves.add((pos[0], pos[1] - 2))
            return valid_moves

        # If king is in check, can only block or capture
        if attack_info.attacking_pieces:
            valid_moves = set()
            king_pos = self.find_king(piece.color)
            attacker, attacker_pos = attack_info.attacking_pieces[0]

            # Can capture the attacker
            if attacker_pos in basic_moves:
                valid_moves.add(attacker_pos)

            # Can block the check (only for sliding piece attacks)
            if attacker.piece_type.is_sliding_piece():
                block_squares = self._get_squares_between(attacker_pos, king_pos)
                valid_moves.update(basic_moves & block_squares)

            return valid_moves

        return basic_moves

    def get_attack_info(self, color: Color) -> AttackInfo:
        """Get information about attacks on the king."""
        king_pos = self.find_king(color)
        if not king_pos:
            return AttackInfo(set(), [], {})

        attacked_squares = set()
        attacking_pieces = []
        pin_lines = {}

        enemy_pieces = self.black_pieces if color == Color.WHITE else self.white_pieces

        # First collect all attacked squares
        for piece, pos in enemy_pieces:
            moves = self.get_basic_moves(pos)
            attacked_squares.update(moves)

            # Only sliding pieces can pin
            if piece.piece_type.is_sliding_piece():
                # Check if piece is aligned with king
                if self._are_aligned(pos, king_pos):
                    # Look for pieces between attacker and king
                    squares_between = self._get_squares_between(pos, king_pos)
                    pieces_between = [
                        (square, self.squares[square[0]][square[1]])
                        for square in squares_between
                        if self.squares[square[0]][square[1]] is not None
                    ]

                    # If exactly one friendly piece is between attacker and king, it's pinned
                    friendly_pieces = [
                        (square, p) for square, p in pieces_between if p.color == color
                    ]
                    if len(pieces_between) == 1 and len(friendly_pieces) == 1:
                        pinned_pos = friendly_pieces[0][0]
                        # Pin line includes attacker position and squares between
                        pin_lines[pinned_pos] = squares_between | {pos}
                    elif len(pieces_between) == 0 and king_pos in moves:
                        # Direct attack on king
                        attacking_pieces.append((piece, pos))
            elif king_pos in moves:
                # Direct attack from non-sliding piece
                attacking_pieces.append((piece, pos))

        return AttackInfo(attacked_squares, attacking_pieces, pin_lines)

    def _are_aligned(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if two positions are aligned (same row, column, or diagonal)."""
        row_diff = pos2[0] - pos1[0]
        col_diff = pos2[1] - pos1[1]

        return (
            row_diff == 0  # Same row
            or col_diff == 0  # Same column
            or abs(row_diff) == abs(col_diff)  # Diagonal
        )

    def _get_ray_between(
        self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]
    ) -> Set[Tuple[int, int]]:
        """Get all squares between two positions if they're aligned (including to_pos, excluding from_pos)."""
        row_diff = to_pos[0] - from_pos[0]
        col_diff = to_pos[1] - from_pos[1]

        # Check if positions are aligned
        if row_diff == 0:  # Same row
            step = (0, 1 if col_diff > 0 else -1)
        elif col_diff == 0:  # Same column
            step = (1 if row_diff > 0 else -1, 0)
        elif abs(row_diff) == abs(col_diff):  # Diagonal
            step = (1 if row_diff > 0 else -1, 1 if col_diff > 0 else -1)
        else:
            return set()  # Not aligned

        squares = set()
        curr = (from_pos[0] + step[0], from_pos[1] + step[1])

        while curr != to_pos:
            squares.add(curr)
            curr = (curr[0] + step[0], curr[1] + step[1])
        squares.add(to_pos)  # Include the target position

        return squares

    def _find_pinned_piece(
        self, from_pos: Tuple[int, int], king_pos: Tuple[int, int], king_color: Color
    ) -> Optional[Tuple[int, int]]:
        """Find a pinned piece between an attacking piece and the king."""
        ray_squares = self._get_ray_to_king(from_pos, king_pos)
        if not ray_squares:
            return None

        found_piece = None
        for square in ray_squares:
            piece = self.squares[square[0]][square[1]]
            if piece:
                if found_piece:  # Found second piece
                    return None
                if piece.color == king_color:
                    found_piece = square
                else:
                    return None

        return found_piece

    def get_basic_moves(self, pos: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Get basic legal moves for a piece without considering check/pins."""
        piece = self.squares[pos[0]][pos[1]]
        if not piece:
            return set()

        moves = set()
        row, col = pos

        # Pawn moves
        if piece.piece_type == PieceType.PAWN:
            direction = 1 if piece.color == Color.WHITE else -1

            # Forward move
            new_row = row + direction
            if 0 <= new_row < 8 and not self.squares[new_row][col]:
                moves.add((new_row, col))
                # Initial two-square move
                if not piece.has_moved:
                    two_forward = row + 2 * direction
                    if 0 <= two_forward < 8 and not self.squares[two_forward][col]:
                        moves.add((two_forward, col))

            # Captures
            for col_offset in [-1, 1]:
                new_col = col + col_offset
                new_row = row + direction
                if 0 <= new_col < 8 and 0 <= new_row < 8:
                    target = self.squares[new_row][new_col]
                    if target and target.color != piece.color:
                        moves.add((new_row, new_col))

        # Knight moves
        elif piece.piece_type == PieceType.KNIGHT:
            offsets = [
                (2, 1),
                (2, -1),
                (-2, 1),
                (-2, -1),
                (1, 2),
                (1, -2),
                (-1, 2),
                (-1, -2),
            ]
            for row_offset, col_offset in offsets:
                new_row, new_col = row + row_offset, col + col_offset
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = self.squares[new_row][new_col]
                    if not target or target.color != piece.color:
                        moves.add((new_row, new_col))

        # King moves
        elif piece.piece_type == PieceType.KING:
            offsets = [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]
            for row_offset, col_offset in offsets:
                new_row, new_col = row + row_offset, col + col_offset
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = self.squares[new_row][new_col]
                    if not target or target.color != piece.color:
                        moves.add((new_row, new_col))

        # Sliding pieces (Bishop, Rook, Queen)
        else:
            directions = []
            if piece.piece_type in {PieceType.BISHOP, PieceType.QUEEN}:
                directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            if piece.piece_type in {PieceType.ROOK, PieceType.QUEEN}:
                directions += [(1, 0), (-1, 0), (0, 1), (0, -1)]

            for row_dir, col_dir in directions:
                new_row, new_col = row + row_dir, col + col_dir
                while 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = self.squares[new_row][new_col]
                    if not target:
                        moves.add((new_row, new_col))
                    else:
                        if target.color != piece.color:
                            moves.add((new_row, new_col))
                        break
                    new_row += row_dir
                    new_col += col_dir

        return moves

    def copy(self):
        """Create a deep copy of the board for move simulation."""
        new_board = object.__new__(Board)  # Create new board without initialization

        # Initialize empty squares
        new_board.squares = [[None for _ in range(8)] for _ in range(8)]
        new_board.white_pieces = []
        new_board.black_pieces = []

        # Copy squares and create new pieces
        for row in range(8):
            for col in range(8):
                piece = self.squares[row][col]
                if piece:
                    new_piece = Piece(piece.piece_type, piece.color)
                    new_piece.has_moved = piece.has_moved
                    new_board.squares[row][col] = new_piece

                    # Add to appropriate piece list
                    if piece.color == Color.WHITE:
                        new_board.white_pieces.append((new_piece, (row, col)))
                    else:
                        new_board.black_pieces.append((new_piece, (row, col)))

        # Copy current turn
        new_board.current_turn = self.current_turn

        return new_board

    def move_piece(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """Execute a move and update piece lists. Returns True if successful."""
        piece = self.squares[from_pos[0]][from_pos[1]]
        if not piece:
            return False

        # Handle captured piece
        captured_piece = self.squares[to_pos[0]][to_pos[1]]
        if captured_piece:
            # Never allow king captures
            if captured_piece.piece_type == PieceType.KING:
                return False

            # Remove captured piece from opponent's piece list
            piece_list = (
                self.black_pieces if piece.color == Color.WHITE else self.white_pieces
            )
            for i, (p, pos) in enumerate(piece_list):
                if pos == to_pos:
                    piece_list.pop(i)
                    break

        # Update board state
        self.squares[to_pos[0]][to_pos[1]] = piece
        self.squares[from_pos[0]][from_pos[1]] = None

        # Update piece lists
        piece_list = (
            self.white_pieces if piece.color == Color.WHITE else self.black_pieces
        )
        for i, (p, pos) in enumerate(piece_list):
            if pos == from_pos:
                piece_list[i] = (piece, to_pos)
                break

        piece.has_moved = True

        # Update current turn
        self.current_turn = (
            Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE
        )

        return True

    def find_king(self, color: Color) -> Optional[Tuple[int, int]]:
        """Find the position of the king of the given color."""
        pieces = self.white_pieces if color == Color.WHITE else self.black_pieces
        for piece, pos in pieces:
            if piece.piece_type == PieceType.KING:
                return pos
        return None

    def is_in_check(self, color: Color) -> bool:
        """Check if the given color's king is in check."""
        king_pos = self.find_king(color)
        return self.is_square_attacked(
            king_pos, Color.BLACK if color == Color.WHITE else Color.WHITE
        )

    def is_checkmate(self, color: Color) -> bool:
        """Check if the given color is in checkmate."""
        # First verify the king is in check
        attack_info = self.get_attack_info(color)
        if not attack_info.attacking_pieces:
            return False

        # Check if any piece has valid moves that get out of check
        pieces = self.white_pieces if color == Color.WHITE else self.black_pieces
        for piece, pos in pieces:
            if self.get_valid_moves(pos):
                return False

        return True

    def is_stalemate(self, color: Color) -> bool:
        """Check if the given color is in stalemate."""
        # If in check, it's not stalemate
        if self.is_in_check(color):
            return False

        # Check if any piece has valid moves
        pieces = self.white_pieces if color == Color.WHITE else self.black_pieces
        for piece, pos in pieces:
            if self.get_valid_moves(pos):
                return False

        return True

    def is_draw(self) -> bool:
        """Check if the game is a draw according to chess rules."""
        # 1. Stalemate
        if self.is_stalemate(Color.WHITE) or self.is_stalemate(Color.BLACK):
            return True

        # 2. Insufficient material
        white_pieces = self.white_pieces
        black_pieces = self.black_pieces

        # Only check insufficient material if both sides have very few pieces
        if len(white_pieces) <= 2 and len(black_pieces) <= 2:
            # King vs King
            if len(white_pieces) == 1 and len(black_pieces) == 1:
                return True

            # King and minor piece vs King
            if (len(white_pieces) == 2 and len(black_pieces) == 1) or (
                len(white_pieces) == 1 and len(black_pieces) == 2
            ):
                for pieces in [white_pieces, black_pieces]:
                    if len(pieces) == 2:
                        non_king_piece = next(
                            p for p, _ in pieces if p.piece_type != PieceType.KING
                        )
                        # Only Knight or Bishop alone is insufficient
                        if non_king_piece.piece_type in {
                            PieceType.BISHOP,
                            PieceType.KNIGHT,
                        }:
                            return True

            # King and Bishop vs King and Bishop (same colored squares)
            if len(white_pieces) == 2 and len(black_pieces) == 2:
                white_bishop = next(
                    (p for p, _ in white_pieces if p.piece_type == PieceType.BISHOP),
                    None,
                )
                black_bishop = next(
                    (p for p, _ in black_pieces if p.piece_type == PieceType.BISHOP),
                    None,
                )
                if white_bishop and black_bishop:
                    white_pos = next(
                        pos
                        for p, pos in white_pieces
                        if p.piece_type == PieceType.BISHOP
                    )
                    black_pos = next(
                        pos
                        for p, pos in black_pieces
                        if p.piece_type == PieceType.BISHOP
                    )
                    if (white_pos[0] + white_pos[1]) % 2 == (
                        black_pos[0] + black_pos[1]
                    ) % 2:
                        return True

        # If we have more pieces, it's not a material-based draw
        return False

    def get_game_result(self) -> float:
        """Get the game result from current player's perspective
        Returns:
            1.0 for win
            -1.0 for loss
            0.0 for draw
        """
        if self.is_checkmate(self.current_turn):
            return -1.0  # Current player is checkmated
        elif self.is_checkmate(
            Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE
        ):
            return 1.0  # Current player delivered checkmate
        elif self.is_draw():
            return 0.0  # Draw

        raise ValueError("Game is not over")

    def _get_squares_between(
        self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]
    ) -> Set[Tuple[int, int]]:
        """Get all squares between two positions (not including the positions themselves)"""
        row_diff = to_pos[0] - from_pos[0]
        col_diff = to_pos[1] - from_pos[1]

        # Must be in same row, column, or diagonal
        if row_diff != 0 and col_diff != 0 and abs(row_diff) != abs(col_diff):
            return set()

        # Determine direction
        row_step = 0 if row_diff == 0 else row_diff // abs(row_diff)
        col_step = 0 if col_diff == 0 else col_diff // abs(col_diff)

        squares = set()
        current = (from_pos[0] + row_step, from_pos[1] + col_step)

        while current != to_pos:
            squares.add(current)
            current = (current[0] + row_step, current[1] + col_step)

        return squares

    def _get_moves_in_check(
        self,
        pos: Tuple[int, int],
        attack_info: AttackInfo,
        basic_moves: Set[Tuple[int, int]],
    ) -> Set[Tuple[int, int]]:
        """Get valid moves when the king is in check."""
        piece = self.squares[pos[0]][pos[1]]
        valid_moves = set()

        # For each possible move, simulate it and check if it resolves the check
        for move in basic_moves:
            board_copy = self.copy()  # Create a copy of the board
            board_copy.make_move(pos, move)
            if not board_copy.get_attack_info(piece.color).attacking_pieces:
                valid_moves.add(move)

        return valid_moves

    def _get_moves_not_in_check(
        self,
        pos: Tuple[int, int],
        attack_info: AttackInfo,
        basic_moves: Set[Tuple[int, int]],
    ) -> Set[Tuple[int, int]]:
        """Get valid moves when not in check, considering pins"""
        piece = self.squares[pos[0]][pos[1]]

        # If piece is pinned, it can only move along the pin line
        if pos in attack_info.pin_lines:
            allowed_moves = basic_moves & attack_info.pin_lines[pos]
            return allowed_moves

        return basic_moves

    def __str__(self) -> str:
        """String representation of the board."""
        # Unicode chess pieces
        piece_symbols = {
            (PieceType.KING, Color.WHITE): "♚",
            (PieceType.QUEEN, Color.WHITE): "♛",
            (PieceType.ROOK, Color.WHITE): "♜",
            (PieceType.BISHOP, Color.WHITE): "♝",
            (PieceType.KNIGHT, Color.WHITE): "♞",
            (PieceType.PAWN, Color.WHITE): "♟",
            (PieceType.KING, Color.BLACK): "♔",
            (PieceType.QUEEN, Color.BLACK): "♕",
            (PieceType.ROOK, Color.BLACK): "♖",
            (PieceType.BISHOP, Color.BLACK): "♗",
            (PieceType.KNIGHT, Color.BLACK): "♘",
            (PieceType.PAWN, Color.BLACK): "♙",
        }

        # Build the board string
        board_str = "  a b c d e f g h\n"
        board_str += "  ---------------\n"

        for row in range(7, -1, -1):  # Print rows from 8 to 1
            board_str += f"{row + 1}|"
            for col in range(8):
                piece = self.squares[row][col]
                if piece:
                    symbol = piece_symbols.get((piece.piece_type, piece.color), "?")
                    board_str += f"{symbol} "
                else:
                    board_str += ". "
            board_str = board_str.rstrip() + "\n"  # Remove trailing space

        return board_str

    def _can_castle_kingside(self, pos: Tuple[int, int], color: Color) -> bool:
        """Check if kingside castling is possible."""
        row, col = pos
        # Check if king and rook are in correct positions and haven't moved
        rook_col = 7
        rook = self.squares[row][rook_col]
        if not rook or rook.piece_type != PieceType.ROOK or rook.has_moved:
            return False

        # Check if squares between king and rook are empty
        for c in range(col + 1, rook_col):
            if self.squares[row][c] is not None:
                return False

        # Check if squares king moves through are not under attack
        attack_info = self.get_attack_info(color)
        for c in range(col, col + 3):  # Include king's current square
            if (row, c) in attack_info.attacked_squares:
                return False

        return True

    def _can_castle_queenside(self, pos: Tuple[int, int], color: Color) -> bool:
        """Check if queenside castling is possible."""
        row, col = pos
        # Check if rook is in correct position and hasn't moved
        rook_col = 0
        rook = self.squares[row][rook_col]
        if not rook or rook.piece_type != PieceType.ROOK or rook.has_moved:
            return False

        # Check if squares between king and rook are empty
        for c in range(rook_col + 1, col):
            if self.squares[row][c] is not None:
                return False

        # Check if squares king moves through are not under attack
        attack_info = self.get_attack_info(color)
        for c in range(col - 2, col + 1):
            if (row, c) in attack_info.attacked_squares:
                return False

        return True

    def get_pieces_by_type(
        self, color: Color, piece_type: PieceType
    ) -> List[Tuple[Piece, Tuple[int, int]]]:
        """Get all pieces of the given type and color."""
        pieces = self.white_pieces if color == Color.WHITE else self.black_pieces
        return [(piece, pos) for piece, pos in pieces if piece.piece_type == piece_type]

    def _move_leaves_in_check(
        self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], color: Color
    ) -> bool:
        # Implement the logic to check if moving from from_pos to to_pos leaves the king in check
        # This is a placeholder and should be implemented based on your specific check detection logic
        return False

    def is_game_over(self) -> bool:
        """Check if the game is over (checkmate or draw)"""
        # Check for checkmate
        if self.is_checkmate(Color.WHITE) or self.is_checkmate(Color.BLACK):
            return True

        # Check for draw
        if self.is_draw():
            return True

        return False

    def is_square_attacked(self, pos: Tuple[int, int], by_color: Color) -> bool:
        """Check if a square is attacked by any piece of the given color."""
        for piece, piece_pos in (
            self.white_pieces if by_color == Color.WHITE else self.black_pieces
        ):
            if pos in self.get_basic_moves(piece_pos):
                return True
        return False
