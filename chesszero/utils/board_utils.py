import mlx.core as mx
from chess_engine.board import Board, Color, PieceType
from chess_engine.bitboard import BitBoard


def encode_board(board: BitBoard) -> mx.array:
    """Convert BitBoard state to network input format"""
    # Convert uint8 to float32 when creating MLX array
    return mx.array(board.state, dtype=mx.float32)


def decode_policy(policy_output: mx.array) -> list:
    """
    Convert policy network output into a list of legal moves with probabilities

    Args:
        policy_output: Network output with shape (4672,)

    Returns:
        List of (move, probability) tuples
    """
    moves = []
    idx = 0

    # Decode each possible move type
    for from_square in range(64):  # 8x8 board
        from_rank = from_square // 8
        from_file = from_square % 8

        # Queen moves (56 planes)
        for direction in range(8):  # 8 directions
            for distance in range(7):  # 7 possible distances
                prob = policy_output[idx]
                if prob > 0:
                    # Convert to chess move
                    moves.append((from_square, direction, distance, prob))
                idx += 1

        # Knight moves (8 planes)
        for knight_move in range(8):
            prob = policy_output[idx]
            if prob > 0:
                moves.append((from_square, "knight", knight_move, prob))
            idx += 1

        # Underpromotions (9 planes)
        for promotion in range(9):
            prob = policy_output[idx]
            if prob > 0:
                moves.append((from_square, "promotion", promotion, prob))
            idx += 1

    return moves
