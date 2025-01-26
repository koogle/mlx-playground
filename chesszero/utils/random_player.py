import random
from chess_engine.board import Board, Color


class RandomPlayer:
    """A player that makes random legal moves"""

    def select_move(self, board: Board):
        """Select a random legal move from all possible moves"""
        pieces = (
            board.white_pieces
            if board.current_turn == Color.WHITE
            else board.black_pieces
        )
        valid_moves = []

        # Collect all valid moves
        for piece, pos in pieces:
            moves = board.get_valid_moves(pos)
            for move in moves:
                valid_moves.append((pos, move))

        if not valid_moves:
            return None

        return random.choice(valid_moves)
