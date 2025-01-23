from enum import Enum
from typing import List, Tuple, Optional


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

    def __str__(self):
        # Unicode chess pieces
        unicode_pieces = {
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
        return unicode_pieces[(self.piece_type, self.color)]


class Board:
    def __init__(self):
        self.squares: List[List[Optional[Piece]]] = [
            [None for _ in range(8)] for _ in range(8)
        ]
        self.initialize_board()

    def initialize_board(self):
        # Initialize pawns
        for col in range(8):
            self.squares[1][col] = Piece(PieceType.PAWN, Color.WHITE)
            self.squares[6][col] = Piece(PieceType.PAWN, Color.BLACK)

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
            self.squares[0][col] = Piece(piece_order[col], Color.WHITE)
            self.squares[7][col] = Piece(piece_order[col], Color.BLACK)

    def __str__(self):
        result = []
        result.append("  a b c d e f g h")
        result.append("  ---------------")
        for row in range(7, -1, -1):
            row_str = f"{row + 1}|"
            for col in range(8):
                piece = self.squares[row][col]
                if piece is None:
                    row_str += "."
                else:
                    row_str += str(piece)
                row_str += " "
            result.append(row_str)
        return "\n".join(result)

    def is_square_under_attack(self, square: Tuple[int, int], by_color: Color) -> bool:
        """Check if a square is under attack by any piece of the given color."""
        row, col = square
        for r in range(8):
            for c in range(8):
                piece = self.squares[r][c]
                if piece and piece.color == by_color:
                    if self.is_valid_move((r, c), square, ignore_turn=True):
                        return True
        return False

    def find_king(self, color: Color) -> Optional[Tuple[int, int]]:
        """Find the position of the king of the given color."""
        for row in range(8):
            for col in range(8):
                piece = self.squares[row][col]
                if (
                    piece
                    and piece.piece_type == PieceType.KING
                    and piece.color == color
                ):
                    return (row, col)
        return None

    def is_in_check(self, color: Color) -> bool:
        """Check if the king of the given color is in check."""
        king_pos = self.find_king(color)
        if not king_pos:
            return False

        opponent_color = Color.BLACK if color == Color.WHITE else Color.WHITE
        return self.is_square_under_attack(king_pos, opponent_color)

    def is_checkmate(self, color: Color) -> bool:
        """Check if the given color is in checkmate."""
        if not self.is_in_check(color):
            return False

        # Try all possible moves for all pieces of this color
        for from_row in range(8):
            for from_col in range(8):
                piece = self.squares[from_row][from_col]
                if not piece or piece.color != color:
                    continue

                for to_row in range(8):
                    for to_col in range(8):
                        if self.is_valid_move((from_row, from_col), (to_row, to_col)):
                            # Try the move
                            original_target = self.squares[to_row][to_col]
                            self.squares[to_row][to_col] = piece
                            self.squares[from_row][from_col] = None

                            # Check if this gets us out of check
                            still_in_check = self.is_in_check(color)

                            # Undo the move
                            self.squares[from_row][from_col] = piece
                            self.squares[to_row][to_col] = original_target

                            if not still_in_check:
                                return False

        return True

    def is_valid_move(
        self,
        from_square: Tuple[int, int],
        to_square: Tuple[int, int],
        ignore_turn: bool = False,
        check_for_check: bool = True,
    ) -> bool:
        """Check if a move is valid according to chess rules."""
        from_row, from_col = from_square
        to_row, to_col = to_square

        # Basic bounds checking
        if not (
            0 <= from_row < 8
            and 0 <= from_col < 8
            and 0 <= to_row < 8
            and 0 <= to_col < 8
        ):
            return False

        piece = self.squares[from_row][from_col]
        if not piece:
            return False

        target = self.squares[to_row][to_col]
        # Can't capture your own piece
        if target and target.color == piece.color:
            return False

        # Get the movement vector
        row_diff = to_row - from_row
        col_diff = to_col - from_col

        # Handle each piece type
        if piece.piece_type == PieceType.PAWN:
            return self._is_valid_pawn_move(from_square, to_square, piece.color)

        # Get the basic movement patterns for this piece
        patterns = piece.piece_type.get_move_patterns()

        if piece.piece_type == PieceType.KNIGHT:
            return (row_diff, col_diff) in patterns

        if piece.piece_type.is_sliding_piece():
            # For sliding pieces, check if movement is along valid direction
            for pattern in patterns:
                pattern_row, pattern_col = pattern
                if row_diff and col_diff:  # Diagonal movement
                    if abs(row_diff) != abs(col_diff):
                        continue
                    step_row = row_diff // abs(row_diff)
                    step_col = col_diff // abs(col_diff)
                else:  # Straight movement
                    if row_diff:
                        step_row = row_diff // abs(row_diff)
                        step_col = 0
                    else:
                        step_row = 0
                        step_col = col_diff // abs(col_diff)

                # Check if path is clear
                curr_row, curr_col = from_row + step_row, from_col + step_col
                while (curr_row, curr_col) != (to_row, to_col):
                    if self.squares[curr_row][curr_col] is not None:
                        return False
                    curr_row += step_row
                    curr_col += step_col
                return True

        if piece.piece_type == PieceType.KING:
            return (row_diff, col_diff) in patterns

        if check_for_check:
            # Try the move and see if it leaves/puts the king in check
            original_target = self.squares[to_row][to_col]
            self.squares[to_row][to_col] = piece
            self.squares[from_row][from_col] = None

            # Check if this move puts/leaves own king in check
            in_check = self.is_in_check(piece.color)

            # Undo the move
            self.squares[from_row][from_col] = piece
            self.squares[to_row][to_col] = original_target

            if in_check:
                return False

        return True

    def _is_valid_pawn_move(
        self, from_square: Tuple[int, int], to_square: Tuple[int, int], color: Color
    ) -> bool:
        """Check if a pawn move is valid."""
        from_row, from_col = from_square
        to_row, to_col = to_square

        direction = 1 if color == Color.WHITE else -1
        start_row = 1 if color == Color.WHITE else 6

        # Normal move forward
        if from_col == to_col and self.squares[to_row][to_col] is None:
            if to_row == from_row + direction:
                return True
            # Initial double move
            if (
                from_row == start_row
                and to_row == from_row + 2 * direction
                and self.squares[from_row + direction][from_col] is None
            ):
                return True

        # Capture move
        if abs(to_col - from_col) == 1 and to_row == from_row + direction:
            target = self.squares[to_row][to_col]
            return target is not None and target.color != color

        return False
