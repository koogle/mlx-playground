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

    def is_square_under_attack(
        self, square: Tuple[int, int], by_color: Color, ignore_king: bool = False
    ) -> bool:
        """Check if a square is under attack by any piece of the given color."""
        row, col = square
        for r in range(8):
            for c in range(8):
                piece = self.squares[r][c]
                if piece and piece.color == by_color:
                    if ignore_king and piece.piece_type == PieceType.KING:
                        continue
                    # Pass check_for_check=False to avoid infinite recursion
                    if self.is_valid_move((r, c), square, check_for_check=False):
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
        is_check = self.is_square_under_attack(king_pos, opponent_color)
        return is_check

    def is_checkmate(self, color: Color) -> bool:
        """Check if the given color is in checkmate."""
        # First verify that the king is in check
        if not self.is_in_check(color):
            return False

        # Get king's position
        king_pos = self.find_king(color)
        if not king_pos:
            return False  # This shouldn't happen in a valid game

        # Try all possible moves for all pieces of this color
        for from_row in range(8):
            for from_col in range(8):
                piece = self.squares[from_row][from_col]
                if not piece or piece.color != color:
                    continue

                # Try all possible destination squares
                for to_row in range(8):
                    for to_col in range(8):
                        # Skip if the move isn't valid (without check validation)
                        if not self.is_valid_move(
                            (from_row, from_col),
                            (to_row, to_col),
                            check_for_check=False,
                        ):
                            continue

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
                            return False  # Found a legal move that escapes check

        return True  # No legal moves found to escape check

    def is_valid_move(
        self,
        from_square: Tuple[int, int],
        to_square: Tuple[int, int],
        check_for_check: bool = True,
    ) -> bool:
        """Check if a move is valid according to chess rules."""
        from_row, from_col = from_square
        to_row, to_col = to_square
        piece = self.squares[from_row][from_col]

        # For debugging pawns
        if piece and piece.piece_type == PieceType.PAWN:
            print(f"\nDebug pawn move from {from_square} to {to_square}")
            print(f"Direction: {'up' if piece.color == Color.WHITE else 'down'}")
            print(
                f"Move type: {'capture' if abs(from_col - to_col) == 1 else 'forward'}"
            )
            if from_col == to_col:  # Normal move
                print(f"Row diff: {to_row - from_row}")
                print(
                    f"Target square occupied: {self.squares[to_row][to_col] is not None}"
                )
            else:  # Capture move
                print(f"Target piece: {self.squares[to_row][to_col]}")

        # Basic bounds checking
        if not (
            0 <= from_row < 8
            and 0 <= from_col < 8
            and 0 <= to_row < 8
            and 0 <= to_col < 8
        ):
            return False

        if not piece:
            return False

        target = self.squares[to_row][to_col]
        # Can't capture your own piece
        if target and target.color == piece.color:
            return False

        # For bishops, rooks, and queens, check if path is clear and move is valid
        if piece.piece_type in {PieceType.BISHOP, PieceType.ROOK, PieceType.QUEEN}:
            # Check if the move is valid for the piece type
            is_diagonal = abs(to_row - from_row) == abs(to_col - from_col)
            is_straight = to_row == from_row or to_col == from_col

            if piece.piece_type == PieceType.BISHOP and not is_diagonal:
                return False
            if piece.piece_type == PieceType.ROOK and not is_straight:
                return False
            if piece.piece_type == PieceType.QUEEN and not (is_diagonal or is_straight):
                return False

            # Check if path is clear
            row_step = (
                0
                if from_row == to_row
                else (to_row - from_row) // abs(to_row - from_row)
            )
            col_step = (
                0
                if from_col == to_col
                else (to_col - from_col) // abs(to_col - from_col)
            )

            current_row, current_col = from_row + row_step, from_col + col_step
            while (current_row, current_col) != (to_row, to_col):
                if self.squares[current_row][current_col] is not None:
                    return False
                current_row += row_step
                current_col += col_step

        # For knights, ensure they move in L-shape
        elif piece.piece_type == PieceType.KNIGHT:
            row_diff = abs(to_row - from_row)
            col_diff = abs(to_col - from_col)
            if not (
                (row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2)
            ):
                return False

        # For king, ensure it moves only one square in any direction
        elif piece.piece_type == PieceType.KING:
            if abs(to_row - from_row) > 1 or abs(to_col - from_col) > 1:
                return False

        # For pawns, ensure they move forward only (except for captures)
        elif piece.piece_type == PieceType.PAWN:
            direction = 1 if piece.color == Color.WHITE else -1
            if from_col == to_col:  # Normal move
                if direction * (to_row - from_row) <= 0:
                    print("Failed: Wrong direction")
                    return False
                if abs(to_row - from_row) > 2:
                    print("Failed: Too far")
                    return False
                if abs(to_row - from_row) == 2:
                    if from_row != (1 if piece.color == Color.WHITE else 6):
                        print("Failed: Double move not from starting position")
                        return False
                    if self.squares[from_row + direction][from_col] is not None:
                        print("Failed: Path blocked for double move")
                        return False
                if self.squares[to_row][to_col] is not None:
                    print("Failed: Target square occupied")
                    return False
            elif abs(from_col - to_col) == 1:  # Capture
                if abs(to_row - from_row) != 1:
                    print("Failed: Invalid capture distance")
                    return False
                if direction * (to_row - from_row) <= 0:
                    print("Failed: Wrong capture direction")
                    return False
                if self.squares[to_row][to_col] is None:
                    print("Failed: No piece to capture")
                    return False

        # Check if move would leave/put own king in check
        if check_for_check:
            # Try the move
            original_target = self.squares[to_row][to_col]
            self.squares[to_row][to_col] = piece
            self.squares[from_row][from_col] = None

            # Check if this move puts/leaves own king in check
            king_pos = self.find_king(piece.color)
            if king_pos:
                opponent_color = (
                    Color.BLACK if piece.color == Color.WHITE else Color.WHITE
                )
                in_check = self.is_square_under_attack(
                    king_pos, opponent_color, ignore_king=True
                )
            else:
                in_check = False

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
