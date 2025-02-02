import random
from chess_engine.bitboard import BitBoard


class RandomPlayer:
    """A player that makes random legal moves using the chess engine's API"""

    def select_move(self, board: BitBoard):
        """
        Select a random legal move using the game's move validation

        Args:
            board: Current board state

        Returns:
            tuple: (from_pos, to_pos) coordinates of the move, or None if no moves available
        """
        # Get all pieces of current color
        current_color = board.get_current_turn()
        pieces = board.get_all_pieces(current_color)

        # Collect all valid moves using list comprehension for better efficiency
        valid_moves = [
            (pos, move) for pos, _ in pieces for move in board.get_valid_moves(pos)
        ]

        if not valid_moves:
            print("\nNo valid moves found!")
            print("Current board state:")
            print(board)
            print(f"Current turn: {'White' if current_color == 0 else 'Black'}")
            return None

        # Select random move from valid moves
        return random.choice(valid_moves)
