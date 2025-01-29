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
            print("\nNo valid moves found!")
            print("Current board state:")
            print(board)
            print(
                f"Current turn: {'White' if board.current_turn == Color.WHITE else 'Black'}"
            )
            return None

        # Keep trying moves until we find one that parses correctly
        while valid_moves:
            # Select random move in algebraic notation
            move_str = random.choice(valid_moves)
            from_pos, to_pos = game._parse_move(move_str)

            if from_pos and to_pos:  # If move parsed successfully
                return (from_pos, to_pos)

            # Remove failed move and try another
            valid_moves.remove(move_str)
            print(f"Failed to parse move: {move_str}")

        print("Failed to parse any moves!")
        print("Available moves were:", valid_moves)
        return None
