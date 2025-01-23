from board import Board, Color, Piece, PieceType
from typing import Tuple, List, Optional


class ChessGame:
    def __init__(self):
        self.board = Board()
        self.current_turn = Color.WHITE
        self.move_history: List[str] = []

    def make_move(self, from_pos: str, to_pos: str) -> bool:
        """
        Make a move using algebraic notation (e.g., 'e2' to 'e4')
        Returns True if the move is valid and was executed
        """
        # Convert algebraic notation to board coordinates
        from_col = ord(from_pos[0].lower()) - ord("a")
        from_row = int(from_pos[1]) - 1
        to_col = ord(to_pos[0].lower()) - ord("a")
        to_row = int(to_pos[1]) - 1

        # Basic validation
        if not (
            0 <= from_col < 8
            and 0 <= from_row < 8
            and 0 <= to_col < 8
            and 0 <= to_row < 8
        ):
            return False

        piece = self.board.squares[from_row][from_col]
        if piece is None or piece.color != self.current_turn:
            return False

        # TODO: Add move validation logic here

        # Execute move
        self.board.squares[to_row][to_col] = piece
        self.board.squares[from_row][from_col] = None

        # Switch turns
        self.current_turn = (
            Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE
        )

        # Record move
        self.move_history.append(f"{from_pos}->{to_pos}")
        return True

    def get_current_turn(self) -> Color:
        return self.current_turn

    def __str__(self):
        return str(self.board)
