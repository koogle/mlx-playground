import numpy as np
from chess_engine.board import Board, Color, PieceType


def encode_board(board: Board) -> np.ndarray:
    """
    Encode the chess board state into the neural network input format

    Returns:
        np.ndarray: Encoded board state with shape (8, 8, 119)
    """
    planes = np.zeros((8, 8, 119), dtype=np.float32)

    # Current position - 6 pieces x 2 colors x 8 history positions = 96 planes
    history_offset = 0
    for piece_type in PieceType:
        for color in Color:
            # Get all pieces of this type and color
            pieces = board.get_pieces_by_type(color, piece_type)
            for piece, pos in pieces:
                planes[pos[0], pos[1], history_offset] = 1
            history_offset += 1

    # Castling rights - 4 planes
    if board.squares[0][4] and not board.squares[0][4].has_moved:  # White king
        if board.squares[0][7] and not board.squares[0][7].has_moved:  # White kingside
            planes[:, :, 96] = 1
        if board.squares[0][0] and not board.squares[0][0].has_moved:  # White queenside
            planes[:, :, 97] = 1

    if board.squares[7][4] and not board.squares[7][4].has_moved:  # Black king
        if board.squares[7][7] and not board.squares[7][7].has_moved:  # Black kingside
            planes[:, :, 98] = 1
        if board.squares[7][0] and not board.squares[7][0].has_moved:  # Black queenside
            planes[:, :, 99] = 1

    # Additional features - 19 planes
    # Color to move
    if board.current_turn == Color.WHITE:
        planes[:, :, 100] = 1

    # Move count and repetition planes would be added here
    # Additional game state planes would be added here

    return planes


def decode_policy(policy_output: np.ndarray) -> list:
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
