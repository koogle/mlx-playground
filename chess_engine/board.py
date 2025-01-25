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
        self.has_moved = False

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

    def get_valid_moves(
        self, pos: Tuple[int, int], board: "Board"
    ) -> List[Tuple[int, int]]:
        """Get all valid moves for this piece from the given position."""
        row, col = pos
        valid_moves = []

        if self.piece_type == PieceType.PAWN:
            direction = 1 if self.color == Color.WHITE else -1
            start_row = 1 if self.color == Color.WHITE else 6

            # Forward moves
            one_forward = row + direction
            if 0 <= one_forward < 8 and board.squares[one_forward][col] is None:
                valid_moves.append((one_forward, col))
                # Initial two-square move
                if not self.has_moved:
                    two_forward = row + 2 * direction
                    if 0 <= two_forward < 8 and board.squares[two_forward][col] is None:
                        valid_moves.append((two_forward, col))

            # Captures
            for col_offset in [-1, 1]:
                new_col = col + col_offset
                if 0 <= new_col < 8 and 0 <= one_forward < 8:
                    target = board.squares[one_forward][new_col]
                    if target and target.color != self.color:
                        valid_moves.append((one_forward, new_col))

        elif self.piece_type == PieceType.KNIGHT:
            for row_offset, col_offset in self.piece_type.get_move_patterns():
                new_row, new_col = row + row_offset, col + col_offset
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = board.squares[new_row][new_col]
                    if not target or target.color != self.color:
                        valid_moves.append((new_row, new_col))

        elif self.piece_type.is_sliding_piece():
            for row_dir, col_dir in self.piece_type.get_move_patterns():
                new_row, new_col = row + row_dir, col + col_dir
                while 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = board.squares[new_row][new_col]
                    if not target:
                        valid_moves.append((new_row, new_col))
                    else:
                        if target.color != self.color:
                            valid_moves.append((new_row, new_col))
                        break
                    new_row += row_dir
                    new_col += col_dir

        elif self.piece_type == PieceType.KING:
            for row_offset, col_offset in self.piece_type.get_move_patterns():
                new_row, new_col = row + row_offset, col + col_offset
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = board.squares[new_row][new_col]
                    if not target or target.color != self.color:
                        valid_moves.append((new_row, new_col))

        return valid_moves


class Board:
    def __init__(self):
        self.squares: List[List[Optional[Piece]]] = [
            [None for _ in range(8)] for _ in range(8)
        ]
        self.white_pieces: List[Tuple[Piece, Tuple[int, int]]] = []  # (piece, position)
        self.black_pieces: List[Tuple[Piece, Tuple[int, int]]] = []
        self.initialize_board()

    def initialize_board(self):
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

    def get_valid_moves(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get all valid moves for a piece at the given position."""
        row, col = pos
        piece = self.squares[row][col]
        if not piece:
            return []

        # Get basic valid moves for the piece
        valid_moves = piece.get_valid_moves(pos, self)

        # Filter moves that would leave king in check
        return [
            move
            for move in valid_moves
            if not self._move_causes_check(pos, move, piece.color)
        ]

    def move_piece(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> None:
        """Execute a move and update piece lists."""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        piece = self.squares[from_row][from_col]

        if not piece:
            return

        # Remove captured piece if any
        captured = self.squares[to_row][to_col]
        if captured:
            piece_list = (
                self.white_pieces
                if captured.color == Color.WHITE
                else self.black_pieces
            )
            self._remove_piece(captured)

        # Update piece position
        self._update_piece_position(piece, from_pos, to_pos)
        piece.has_moved = True

    def get_pieces(self, color: Color) -> List[Tuple[Piece, Tuple[int, int]]]:
        """Get all pieces of the given color with their positions."""
        return self.white_pieces if color == Color.WHITE else self.black_pieces

    def get_pieces_by_type(
        self, color: Color, piece_type: PieceType
    ) -> List[Tuple[Piece, Tuple[int, int]]]:
        """Get all pieces of the given color and type with their positions."""
        pieces = self.get_pieces(color)
        return [(piece, pos) for piece, pos in pieces if piece.piece_type == piece_type]

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

        # Basic bounds checking
        if not (
            0 <= from_row < 8
            and 0 <= from_col < 8
            and 0 <= to_row < 8
            and 0 <= to_col < 8
        ):
            print(f"Move {from_square} to {to_square} out of bounds")
            return False

        if not piece:
            print(f"No piece at {from_square}")
            return False

        # For debugging pawns
        if piece and piece.piece_type == PieceType.PAWN:
            print(
                f"\nDebug pawn move from ({from_row},{from_col}) to ({to_row},{to_col})"
            )
            print(f"Direction: {'up' if piece.color == Color.WHITE else 'down'}")
            print(f"Move distance: rows={to_row - from_row}, cols={to_col - from_col}")
            print(f"Target square: {self.squares[to_row][to_col]}")

        target = self.squares[to_row][to_col]
        # Can't capture your own piece
        if target and target.color == piece.color:
            print(f"Can't capture own piece at {to_square}")
            return False

        # For pawns, ensure they move forward only (except for captures)
        if piece.piece_type == PieceType.PAWN:
            direction = 1 if piece.color == Color.WHITE else -1

            # Check if moving in correct direction
            if (to_row - from_row) * direction <= 0:
                print(f"Pawn must move {direction} rows forward")
                return False

            if from_col == to_col:  # Normal move
                # Check distance
                if abs(to_row - from_row) > 2:
                    print("Pawn can only move 1 or 2 squares forward")
                    return False

                # Check if it's a valid double move
                if abs(to_row - from_row) == 2:
                    if from_row != (1 if piece.color == Color.WHITE else 6):
                        print("Double move only allowed from starting position")
                        return False
                    # Check if path is clear
                    if self.squares[from_row + direction][from_col] is not None:
                        print("Path blocked for double move")
                        return False

                # Check if target square is empty
                if self.squares[to_row][to_col] is not None:
                    print("Cannot move forward into occupied square")
                    return False

            elif abs(to_col - from_col) == 1:  # Capture move
                if abs(to_row - from_row) != 1:
                    print("Diagonal capture must move exactly one square")
                    return False
                if self.squares[to_row][to_col] is None:
                    print("No piece to capture diagonally")
                    return False
                if self.squares[to_row][to_col].color == piece.color:
                    print("Cannot capture own piece")
                    return False
            else:
                print("Invalid pawn move")
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

    def get_piece_moves(
        self, from_square: Tuple[int, int], check_for_check: bool = True
    ) -> List[Tuple[int, int]]:
        """Get all valid moves for a piece at the given square."""
        from_row, from_col = from_square
        piece = self.squares[from_row][from_col]
        if not piece:
            return []

        valid_moves = []

        if piece.piece_type == PieceType.PAWN:
            # Pawn moves
            direction = 1 if piece.color == Color.WHITE else -1
            start_row = 1 if piece.color == Color.WHITE else 6

            # Forward moves
            one_forward = from_row + direction
            if 0 <= one_forward < 8 and self.squares[one_forward][from_col] is None:
                valid_moves.append((one_forward, from_col))
                # Initial two-square move
                if from_row == start_row:
                    two_forward = from_row + 2 * direction
                    if self.squares[two_forward][from_col] is None:
                        valid_moves.append((two_forward, from_col))

            # Captures
            for col_offset in [-1, 1]:
                new_col = from_col + col_offset
                if 0 <= new_col < 8:
                    target_square = (one_forward, new_col)
                    target = self.squares[one_forward][new_col]
                    if target and target.color != piece.color:
                        valid_moves.append(target_square)

        elif piece.piece_type == PieceType.KNIGHT:
            # Knight moves
            for row_offset, col_offset in [
                (2, 1),
                (2, -1),
                (-2, 1),
                (-2, -1),
                (1, 2),
                (1, -2),
                (-1, 2),
                (-1, -2),
            ]:
                new_row, new_col = from_row + row_offset, from_col + col_offset
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = self.squares[new_row][new_col]
                    if not target or target.color != piece.color:
                        valid_moves.append((new_row, new_col))

        elif piece.piece_type in {PieceType.BISHOP, PieceType.ROOK, PieceType.QUEEN}:
            # Sliding piece moves
            directions = []
            if piece.piece_type in {PieceType.BISHOP, PieceType.QUEEN}:
                directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
            if piece.piece_type in {PieceType.ROOK, PieceType.QUEEN}:
                directions.extend([(0, 1), (0, -1), (1, 0), (-1, 0)])

            for row_dir, col_dir in directions:
                new_row, new_col = from_row + row_dir, from_col + col_dir
                while 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = self.squares[new_row][new_col]
                    if not target:
                        valid_moves.append((new_row, new_col))
                    else:
                        if target.color != piece.color:
                            valid_moves.append((new_row, new_col))
                        break
                    new_row += row_dir
                    new_col += col_dir

        elif piece.piece_type == PieceType.KING:
            # King moves
            for row_offset in [-1, 0, 1]:
                for col_offset in [-1, 0, 1]:
                    if row_offset == 0 and col_offset == 0:
                        continue
                    new_row, new_col = from_row + row_offset, from_col + col_offset
                    if 0 <= new_row < 8 and 0 <= new_col < 8:
                        target = self.squares[new_row][new_col]
                        if not target or target.color != piece.color:
                            valid_moves.append((new_row, new_col))

        # Filter moves that would leave king in check
        if check_for_check:
            valid_moves = [
                move
                for move in valid_moves
                if not self._move_causes_check(from_square, move, piece.color)
            ]

        return valid_moves

    def _move_causes_check(
        self, from_square: Tuple[int, int], to_square: Tuple[int, int], color: Color
    ) -> bool:
        """Check if a move would leave the king in check."""
        from_row, from_col = from_square
        to_row, to_col = to_square
        piece = self.squares[from_row][from_col]

        # Try the move
        original_target = self.squares[to_row][to_col]
        self.squares[to_row][to_col] = piece
        self.squares[from_row][from_col] = None

        # Check if this move puts/leaves own king in check
        king_pos = self.find_king(color)
        in_check = False
        if king_pos:
            opponent_color = Color.BLACK if color == Color.WHITE else Color.WHITE
            in_check = self.is_square_under_attack(
                king_pos, opponent_color, ignore_king=True
            )

        # Undo the move
        self.squares[from_row][from_col] = piece
        self.squares[to_row][to_col] = original_target

        return in_check

    def _remove_piece(self, piece: Piece):
        # Implementation of _remove_piece method
        pass

    def _update_piece_position(
        self, piece: Piece, from_pos: Tuple[int, int], to_pos: Tuple[int, int]
    ):
        # Implementation of _update_piece_position method
        pass
