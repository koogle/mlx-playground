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
                if 0 <= new_col < 8 and 0 <= one_forward < 8:  # Check board boundaries
                    target = board.squares[one_forward][new_col]
                    if target and target.color != self.color:  # Enemy piece present
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
            # Normal king moves
            for row_offset, col_offset in self.piece_type.get_move_patterns():
                new_row, new_col = row + row_offset, col + col_offset
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = board.squares[new_row][new_col]
                    if not target or target.color != self.color:
                        valid_moves.append((new_row, new_col))

            # Add castling moves if conditions are met
            if not self.has_moved and not board.is_in_check(self.color):
                # Check kingside castling
                if self._can_castle_kingside(pos, board):
                    valid_moves.append((row, col + 2))
                # Check queenside castling
                if self._can_castle_queenside(pos, board):
                    valid_moves.append((row, col - 2))

        return valid_moves

    def _can_castle_kingside(self, pos: Tuple[int, int], board: "Board") -> bool:
        """Check if kingside castling is possible."""
        row, col = pos
        # Check if rook is in correct position and hasn't moved
        rook_col = 7
        rook = board.squares[row][rook_col]
        if not rook or rook.piece_type != PieceType.ROOK or rook.has_moved:
            return False

        # Check if squares between king and rook are empty
        for c in range(col + 1, rook_col):
            if board.squares[row][c] is not None:
                return False

        # Check if squares king moves through are not under attack
        opponent_color = Color.BLACK if self.color == Color.WHITE else Color.WHITE
        for c in range(col, col + 3):
            if board.is_square_under_attack((row, c), opponent_color):
                return False

        return True

    def _can_castle_queenside(self, pos: Tuple[int, int], board: "Board") -> bool:
        """Check if queenside castling is possible."""
        row, col = pos
        # Check if rook is in correct position and hasn't moved
        rook_col = 0
        rook = board.squares[row][rook_col]
        if not rook or rook.piece_type != PieceType.ROOK or rook.has_moved:
            return False

        # Check if squares between king and rook are empty
        for c in range(rook_col + 1, col):
            if board.squares[row][c] is not None:
                return False

        # Check if squares king moves through are not under attack
        opponent_color = Color.BLACK if self.color == Color.WHITE else Color.WHITE
        for c in range(col - 2, col + 1):
            if board.is_square_under_attack((row, c), opponent_color):
                return False

        return True


class Board:
    DEBUG = False  # Class variable to control debug output

    def __init__(self):
        self.squares: List[List[Optional[Piece]]] = [
            [None for _ in range(8)] for _ in range(8)
        ]
        self.white_pieces: List[Tuple[Piece, Tuple[int, int]]] = []
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

    def get_valid_moves(
        self, pos: Tuple[int, int], check_for_check: bool = True
    ) -> List[Tuple[int, int]]:
        """Get all valid moves for a piece at the given position."""
        row, col = pos
        piece = self.squares[row][col]
        if not piece:
            return []

        # Get basic valid moves for the piece
        valid_moves = piece.get_valid_moves(pos, self)

        if self.DEBUG and piece.piece_type == PieceType.KING:
            print(f"Basic king moves from {pos}: {valid_moves}")

        # Filter moves that would leave king in check
        if check_for_check:
            filtered_moves = []
            for move in valid_moves:
                # Make temporary move
                original_target = self.squares[move[0]][move[1]]
                self.squares[move[0]][move[1]] = piece
                self.squares[pos[0]][pos[1]] = None

                # Check if king would be in check
                if not self.is_in_check(piece.color):
                    filtered_moves.append(move)
                elif self.DEBUG:
                    print(f"Debug: Move from {pos} to {move} causes check")
                    target_piece = self.squares[move[0]][move[1]]
                    if target_piece:
                        print(
                            f"Target square contains: {target_piece.piece_type} of color {target_piece.color}"
                        )
                    attacking_pieces = self._get_attacking_pieces(
                        move, Color.WHITE if piece.color == Color.BLACK else Color.WHITE
                    )
                    print(f"Square {move} is attacked by: {attacking_pieces}")

                # Undo temporary move
                self.squares[pos[0]][pos[1]] = piece
                self.squares[move[0]][move[1]] = original_target

            valid_moves = filtered_moves

        return valid_moves

    def _get_attacking_pieces(
        self, square: Tuple[int, int], color: Color
    ) -> List[Tuple[Piece, Tuple[int, int]]]:
        """Get all pieces of given color that attack a square."""
        attacking_pieces = []
        pieces = self.white_pieces if color == Color.WHITE else self.black_pieces

        for piece, pos in pieces:
            if square in piece.get_valid_moves(pos, self):
                attacking_pieces.append((piece.piece_type, pos))

        return attacking_pieces

    def move_piece(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> None:
        """Execute a move and update piece lists."""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        piece = self.squares[from_row][from_col]

        if not piece:
            return

        # Handle castling moves
        if piece.piece_type == PieceType.KING and abs(to_col - from_col) == 2:
            # Kingside castling
            if to_col > from_col:
                rook_from = (from_row, 7)
                rook_to = (from_row, to_col - 1)
            # Queenside castling
            else:
                rook_from = (from_row, 0)
                rook_to = (from_row, to_col + 1)

            # Move the rook
            rook = self.squares[rook_from[0]][rook_from[1]]
            self.squares[rook_to[0]][rook_to[1]] = rook
            self.squares[rook_from[0]][rook_from[1]] = None

            # Update rook position in piece list
            piece_list = (
                self.white_pieces if piece.color == Color.WHITE else self.black_pieces
            )
            piece_list.remove((rook, rook_from))
            piece_list.append((rook, rook_to))
            rook.has_moved = True

        # Remove captured piece if any
        captured = self.squares[to_row][to_col]
        if captured:
            piece_list = (
                self.white_pieces
                if captured.color == Color.WHITE
                else self.black_pieces
            )
            piece_list.remove((captured, (to_row, to_col)))

        # Update piece position in the appropriate list
        piece_list = (
            self.white_pieces if piece.color == Color.WHITE else self.black_pieces
        )
        piece_list.remove((piece, from_pos))
        piece_list.append((piece, to_pos))

        # Update board
        self.squares[to_row][to_col] = piece
        self.squares[from_row][from_col] = None
        piece.has_moved = True

        # Handle pawn promotion
        if piece.piece_type == PieceType.PAWN and (
            (piece.color == Color.WHITE and to_row == 7)
            or (piece.color == Color.BLACK and to_row == 0)
        ):
            # Promote to queen by default
            new_piece = Piece(PieceType.QUEEN, piece.color)
            piece_list = (
                self.white_pieces if piece.color == Color.WHITE else self.black_pieces
            )
            piece_list.remove((piece, to_pos))
            piece_list.append((new_piece, to_pos))
            self.squares[to_row][to_col] = new_piece

    def find_king(self, color: Color) -> Optional[Tuple[int, int]]:
        """Find the position of the king of the given color."""
        pieces = self.white_pieces if color == Color.WHITE else self.black_pieces
        for piece, pos in pieces:
            if piece.piece_type == PieceType.KING:
                return pos
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
        # First verify the king is in check
        if not self.is_in_check(color):
            return False

        # Get all pieces of the current color
        pieces = self.white_pieces if color == Color.WHITE else self.black_pieces

        # Check if any piece has a legal move
        for piece, pos in pieces:
            valid_moves = self.get_valid_moves(pos)
            if valid_moves:
                return False

        # No legal moves and in check = checkmate
        return True

    def is_stalemate(self, color: Color) -> bool:
        """Check if the given color is in stalemate."""
        # First verify the king is NOT in check
        if self.is_in_check(color):
            return False

        # Find the king
        king_pos = self.find_king(color)
        if not king_pos:
            return False

        # Get all valid moves for the king
        king = self.squares[king_pos[0]][king_pos[1]]
        valid_king_moves = self.get_valid_moves(king_pos)

        # If king has no valid moves but is not in check, check if any other piece can move
        if not valid_king_moves:
            # Get all pieces of the current color except king
            pieces = self.white_pieces if color == Color.WHITE else self.black_pieces
            for piece, pos in pieces:
                if piece.piece_type != PieceType.KING:
                    if self.get_valid_moves(pos):
                        # Some piece can move
                        return False
            # No piece can move and king has no moves but is not in check - it's stalemate
            return True

        # King has valid moves, not stalemate
        return False

    def is_draw(self) -> bool:
        """Check if the game is a draw according to chess rules."""
        # 1. Stalemate
        if self.is_stalemate(Color.WHITE) or self.is_stalemate(Color.BLACK):
            return True

        # 2. Insufficient material
        white_pieces = self.white_pieces
        black_pieces = self.black_pieces

        # King vs King
        if len(white_pieces) == 1 and len(black_pieces) == 1:
            return True

        # King and Bishop/Knight vs King
        if (len(white_pieces) == 2 and len(black_pieces) == 1) or (
            len(white_pieces) == 1 and len(black_pieces) == 2
        ):
            for pieces in [white_pieces, black_pieces]:
                if len(pieces) == 2:
                    piece_type = (
                        pieces[0][0].piece_type
                        if pieces[0][0].piece_type != PieceType.KING
                        else pieces[1][0].piece_type
                    )
                    if piece_type in {PieceType.BISHOP, PieceType.KNIGHT}:
                        return True

        # King and Bishop vs King and Bishop (same colored squares)
        if len(white_pieces) == 2 and len(black_pieces) == 2:
            white_bishop = next(
                (p for p, _ in white_pieces if p.piece_type == PieceType.BISHOP), None
            )
            black_bishop = next(
                (p for p, _ in black_pieces if p.piece_type == PieceType.BISHOP), None
            )
            if white_bishop and black_bishop:
                # Check if bishops are on same colored squares
                white_pos = next(
                    pos for p, pos in white_pieces if p.piece_type == PieceType.BISHOP
                )
                black_pos = next(
                    pos for p, pos in black_pieces if p.piece_type == PieceType.BISHOP
                )
                if (white_pos[0] + white_pos[1]) % 2 == (
                    black_pos[0] + black_pos[1]
                ) % 2:
                    return True

        return False

    def is_square_under_attack(
        self, square: Tuple[int, int], by_color: Color, ignore_king: bool = False
    ) -> bool:
        """Check if a square is under attack by any piece of the given color."""
        target_row, target_col = square
        pieces = self.white_pieces if by_color == Color.WHITE else self.black_pieces

        for piece, (row, col) in pieces:
            if ignore_king and piece.piece_type == PieceType.KING:
                continue

            # Pawn attacks
            if piece.piece_type == PieceType.PAWN:
                direction = 1 if piece.color == Color.WHITE else -1
                attack_row = row + direction
                if attack_row == target_row:
                    if abs(col - target_col) == 1:
                        return True

            # Knight attacks
            elif piece.piece_type == PieceType.KNIGHT:
                row_diff = abs(row - target_row)
                col_diff = abs(col - target_col)
                if (row_diff == 2 and col_diff == 1) or (
                    row_diff == 1 and col_diff == 2
                ):
                    return True

            # Sliding pieces (Bishop, Rook, Queen)
            elif piece.piece_type.is_sliding_piece():
                # Get the direction from the piece to the target
                row_dir = (
                    0
                    if row == target_row
                    else (target_row - row) // abs(target_row - row)
                )
                col_dir = (
                    0
                    if col == target_col
                    else (target_col - col) // abs(target_col - col)
                )

                # Check if the direction is valid for this piece
                if piece.piece_type == PieceType.ROOK and row_dir != 0 and col_dir != 0:
                    continue
                if piece.piece_type == PieceType.BISHOP and (
                    row_dir == 0 or col_dir == 0
                ):
                    continue

                # Check if path is clear
                curr_row, curr_col = row + row_dir, col + col_dir
                path_clear = True
                while curr_row != target_row or curr_col != target_col:
                    if not (0 <= curr_row < 8 and 0 <= curr_col < 8):
                        path_clear = False
                        break
                    if self.squares[curr_row][curr_col] is not None:
                        path_clear = False
                        break
                    curr_row += row_dir
                    curr_col += col_dir
                if path_clear:
                    return True

            # King attacks
            elif piece.piece_type == PieceType.KING:
                if abs(row - target_row) <= 1 and abs(col - target_col) <= 1:
                    return True

        return False

    def _move_causes_check(
        self, from_square: Tuple[int, int], to_square: Tuple[int, int], color: Color
    ) -> bool:
        """Check if a move would leave the king in check."""
        from_row, from_col = from_square
        to_row, to_col = to_square
        piece = self.squares[from_row][from_col]

        # Store original state
        original_target = self.squares[to_row][to_col]
        original_piece_list = (
            self.white_pieces.copy()
            if color == Color.WHITE
            else self.black_pieces.copy()
        )
        original_opponent_list = (
            self.black_pieces.copy()
            if color == Color.WHITE
            else self.white_pieces.copy()
        )

        # Try the move
        self.squares[to_row][to_col] = piece
        self.squares[from_row][from_col] = None

        # Update piece lists
        piece_list = self.white_pieces if color == Color.WHITE else self.black_pieces
        piece_list.remove((piece, from_square))
        piece_list.append((piece, to_square))

        # If capturing, remove captured piece from opponent's list BEFORE checking for check
        opponent_list = self.black_pieces if color == Color.WHITE else self.white_pieces
        if original_target:
            opponent_list.remove((original_target, to_square))

        # Check if this move puts/leaves own king in check
        king_pos = self.find_king(color)
        in_check = False
        if king_pos:
            opponent_color = Color.BLACK if color == Color.WHITE else Color.WHITE
            in_check = self.is_square_under_attack(king_pos, opponent_color)

        # Restore original state
        self.squares[from_row][from_col] = piece
        self.squares[to_row][to_col] = original_target
        if color == Color.WHITE:
            self.white_pieces = original_piece_list
            self.black_pieces = original_opponent_list
        else:
            self.black_pieces = original_piece_list
            self.white_pieces = original_opponent_list

        return in_check

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

    def get_pieces(self, color: Color) -> List[Tuple[Piece, Tuple[int, int]]]:
        """Get all pieces of the given color with their positions."""
        return self.white_pieces if color == Color.WHITE else self.black_pieces

    def get_pieces_by_type(
        self, color: Color, piece_type: PieceType
    ) -> List[Tuple[Piece, Tuple[int, int]]]:
        """Get all pieces of the given color and type with their positions."""
        pieces = self.get_pieces(color)
        return [(piece, pos) for piece, pos in pieces if piece.piece_type == piece_type]

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
        """Get all valid moves for a piece at the given position."""
        row, col = from_square
        piece = self.squares[row][col]
        if not piece:
            return []

        valid_moves = []

        if piece.piece_type == PieceType.PAWN:
            direction = 1 if piece.color == Color.WHITE else -1
            one_forward = row + direction

            # Check forward moves only if within bounds
            if 0 <= one_forward < 8:
                if self.squares[one_forward][col] is None:
                    valid_moves.append((one_forward, col))
                    # Initial two-square move
                    if not piece.has_moved:
                        two_forward = row + 2 * direction
                        if (
                            0 <= two_forward < 8
                            and self.squares[two_forward][col] is None
                        ):
                            valid_moves.append((two_forward, col))

                # Captures
                for col_offset in [-1, 1]:
                    new_col = col + col_offset
                    if 0 <= new_col < 8:  # Check column bounds
                        target = self.squares[one_forward][new_col]
                        if target and target.color != piece.color:
                            valid_moves.append((one_forward, new_col))

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
                new_row, new_col = row + row_offset, col + col_offset
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
                new_row, new_col = row + row_dir, col + col_dir
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
                    new_row, new_col = row + row_offset, col + col_offset
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

    def _remove_piece(self, piece: Piece):
        # Implementation of _remove_piece method
        pass

    def _update_piece_position(
        self, piece: Piece, from_pos: Tuple[int, int], to_pos: Tuple[int, int]
    ):
        # Implementation of _update_piece_position method
        pass
