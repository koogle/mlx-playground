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
    DEBUG = True  # Class variable to control debug output

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

    def get_basic_moves(
        self, pos: Tuple[int, int], ignore_castling: bool = False
    ) -> Set[Tuple[int, int]]:
        """Get all possible moves for a piece without considering check."""
        piece = self.squares[pos[0]][pos[1]]
        if not piece:
            return set()

        moves = set()
        row, col = pos

        if piece.piece_type == PieceType.PAWN:
            direction = 1 if piece.color == Color.WHITE else -1

            # Forward moves
            one_forward = row + direction
            if 0 <= one_forward < 8 and self.squares[one_forward][col] is None:
                moves.add((one_forward, col))
                # Initial two-square move
                if not piece.has_moved:
                    two_forward = row + 2 * direction
                    if 0 <= two_forward < 8 and self.squares[two_forward][col] is None:
                        moves.add((two_forward, col))

            # Captures
            for col_offset in [-1, 1]:
                new_col = col + col_offset
                if 0 <= new_col < 8 and 0 <= one_forward < 8:  # Check board boundaries
                    target = self.squares[one_forward][new_col]
                    if target and target.color != piece.color:  # Enemy piece present
                        moves.add((one_forward, new_col))

        elif piece.piece_type == PieceType.KNIGHT:
            for row_offset, col_offset in piece.piece_type.get_move_patterns():
                new_row, new_col = row + row_offset, col + col_offset
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = self.squares[new_row][new_col]
                    if not target or target.color != piece.color:
                        moves.add((new_row, new_col))

        elif piece.piece_type.is_sliding_piece():
            for row_dir, col_dir in piece.piece_type.get_move_patterns():
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

        elif piece.piece_type == PieceType.KING:
            # Normal moves - just get basic moves without checking for attacks
            for offset in piece.piece_type.get_move_patterns():
                new_row, new_col = row + offset[0], col + offset[1]
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = self.squares[new_row][new_col]
                    if not target or target.color != piece.color:
                        moves.add((new_row, new_col))

            # Castling - only check if not getting basic attack moves
            if not ignore_castling and not piece.has_moved:
                if self._can_castle_kingside(pos, piece.color):
                    moves.add((row, col + 2))
                if self._can_castle_queenside(pos, piece.color):
                    moves.add((row, col - 2))

        return moves

    def get_attack_info(self, color: Color) -> AttackInfo:
        """Get all squares attacked by the opponent and pieces attacking the king."""
        opponent_pieces = (
            self.black_pieces if color == Color.WHITE else self.white_pieces
        )

        attacked_squares = set()
        attacking_pieces = []
        pin_lines = {}

        king_pos = self.find_king(color)
        if not king_pos:
            return AttackInfo(attacked_squares, attacking_pieces, pin_lines)

        # Get all attacked squares and find pieces attacking the king
        for piece, pos in opponent_pieces:
            # Use ignore_castling=True to break recursion
            piece_attacks = self.get_basic_moves(pos, ignore_castling=True)
            attacked_squares.update(piece_attacks)

            if king_pos in piece_attacks:
                attacking_pieces.append((piece, pos))

        # Find pin lines
        self._find_pin_lines(king_pos, color, pin_lines)

        return AttackInfo(attacked_squares, attacking_pieces, pin_lines)

    def get_valid_moves(self, pos: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Get all valid moves for a piece considering check."""
        piece = self.squares[pos[0]][pos[1]]
        if not piece:
            return set()

        # Get attack information
        attack_info = self.get_attack_info(piece.color)
        basic_moves = self.get_basic_moves(pos)

        # For kings, filter out moves to attacked squares
        if piece.piece_type == PieceType.KING:
            basic_moves = {
                move for move in basic_moves if move not in attack_info.attacked_squares
            }

        # If king is in check
        if attack_info.attacking_pieces:
            return self._get_moves_in_check(pos, attack_info, basic_moves)

        # Normal case - not in check
        return self._get_moves_not_in_check(pos, attack_info, basic_moves)

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
        # If piece is pinned, it can only move along the pin line
        if pos in attack_info.pin_lines:
            return basic_moves & attack_info.pin_lines[pos]

        return basic_moves

    def copy(self) -> "Board":
        """Create a deep copy of the board for move simulation."""
        new_board = Board.__new__(Board)  # Create new board without initialization
        new_board.squares = [[None for _ in range(8)] for _ in range(8)]

        # Copy pieces and their positions
        new_board.white_pieces = []
        new_board.black_pieces = []

        for row in range(8):
            for col in range(8):
                piece = self.squares[row][col]
                if piece:
                    # Create new piece with same attributes
                    new_piece = Piece(piece.piece_type, piece.color)
                    new_piece.has_moved = piece.has_moved
                    new_board.squares[row][col] = new_piece

                    # Add to appropriate piece list
                    piece_list = (
                        new_board.white_pieces
                        if piece.color == Color.WHITE
                        else new_board.black_pieces
                    )
                    piece_list.append((new_piece, (row, col)))

        return new_board

    def move_piece(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """Execute a move and update piece lists. Returns True if successful."""
        piece = self.squares[from_pos[0]][from_pos[1]]
        if not piece:
            if self.DEBUG:
                print(f"DEBUG: No piece at {from_pos}")
            return False

        # Verify the move is valid
        valid_moves = self.get_valid_moves(from_pos)
        if to_pos not in valid_moves:
            if self.DEBUG:
                print(f"DEBUG: Move to {to_pos} not in valid moves: {valid_moves}")
                print(f"DEBUG: Piece: {piece.piece_type} at {from_pos}")
            return False

        # Handle castling moves
        if piece.piece_type == PieceType.KING and abs(to_pos[1] - from_pos[1]) == 2:
            row = from_pos[0]
            # Kingside castling
            if to_pos[1] > from_pos[1]:
                if not self.make_move((row, 7), (row, to_pos[1] - 1)):  # Move rook
                    return False
            # Queenside castling
            else:
                if not self.make_move((row, 0), (row, to_pos[1] + 1)):  # Move rook
                    return False

        # Make the main move
        if not self.make_move(from_pos, to_pos):
            return False

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
        attack_info = self.get_attack_info(color)
        return len(attack_info.attacking_pieces) > 0

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
                if self.DEBUG:
                    print(f"DEBUG: {piece.piece_type} at {pos} has valid moves")
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

    def _find_pin_lines(
        self,
        king_pos: Tuple[int, int],
        color: Color,
        pin_lines: Dict[Tuple[int, int], Set[Tuple[int, int]]],
    ):
        king_row, king_col = king_pos

        # Check all 8 directions for pins
        directions = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),  # Rook directions
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),  # Bishop directions
        ]

        for row_dir, col_dir in directions:
            pinned_piece = None
            pinned_pos = None
            pin_line = set()

            new_row, new_col = king_row + row_dir, king_col + col_dir
            while 0 <= new_row < 8 and 0 <= new_col < 8:
                pin_line.add((new_row, new_col))
                current = self.squares[new_row][new_col]

                if current:
                    if current.color == color:
                        if pinned_piece:  # Second piece of same color - no pin
                            pinned_piece = None
                            break
                        pinned_piece = current
                        pinned_pos = (new_row, new_col)
                    else:  # Opponent's piece
                        if pinned_piece:
                            # Check if this piece can pin
                            if (row_dir, col_dir) in [
                                (0, 1),
                                (0, -1),
                                (1, 0),
                                (-1, 0),
                            ] and current.piece_type in {
                                PieceType.ROOK,
                                PieceType.QUEEN,
                            }:
                                pin_lines[pinned_pos] = pin_line
                            elif (row_dir, col_dir) in [
                                (1, 1),
                                (1, -1),
                                (-1, 1),
                                (-1, -1),
                            ] and current.piece_type in {
                                PieceType.BISHOP,
                                PieceType.QUEEN,
                            }:
                                pin_lines[pinned_pos] = pin_line
                        break
                new_row += row_dir
                new_col += col_dir

    def make_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """Execute a move on the board."""
        piece = self.squares[from_pos[0]][from_pos[1]]
        if not piece:
            return False

        # Update board state
        self.squares[to_pos[0]][to_pos[1]] = piece
        self.squares[from_pos[0]][from_pos[1]] = None

        # Update piece lists
        piece_list = (
            self.white_pieces if piece.color == Color.WHITE else self.black_pieces
        )
        piece_list.remove((piece, from_pos))
        piece_list.append((piece, to_pos))

        piece.has_moved = True

        return True

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
        # Check if rook is in correct position and hasn't moved
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
        for c in range(col, col + 3):
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
