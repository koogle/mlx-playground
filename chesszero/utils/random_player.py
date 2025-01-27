import random
from chess_engine.board import Board, Color
from chess_engine.game import ChessGame


class RandomPlayer:
    """A player that makes random legal moves using the chess engine's API"""

    def select_move(self, board: Board):
        """
        Select a random legal move using the game's move validation

        Args:
            board: Current board state

        Returns:
            str: Move in algebraic notation, or None if no moves available
        """
        # Create a temporary game to use its move generation
        game = ChessGame()
        game.board = board
        game.current_turn = board.current_turn

        # Get all valid moves in algebraic notation
        valid_moves = game.get_all_valid_moves()

        if not valid_moves:
            return None

        # Select random move in algebraic notation
        move_str = random.choice(valid_moves)

        # Parse the move back into coordinates
        from_pos, to_pos = game._parse_move(move_str)

        return (from_pos, to_pos) if from_pos and to_pos else None
