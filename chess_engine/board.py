from enum import Enum
from typing import List, Tuple, Optional


class PieceType(Enum):
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6


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
            (PieceType.KING, Color.WHITE): "♔",
            (PieceType.QUEEN, Color.WHITE): "♕",
            (PieceType.ROOK, Color.WHITE): "♖",
            (PieceType.BISHOP, Color.WHITE): "♗",
            (PieceType.KNIGHT, Color.WHITE): "♘",
            (PieceType.PAWN, Color.WHITE): "♙",
            (PieceType.KING, Color.BLACK): "♚",
            (PieceType.QUEEN, Color.BLACK): "♛",
            (PieceType.ROOK, Color.BLACK): "♜",
            (PieceType.BISHOP, Color.BLACK): "♝",
            (PieceType.KNIGHT, Color.BLACK): "♞",
            (PieceType.PAWN, Color.BLACK): "♟",
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
