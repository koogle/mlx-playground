import mlx.core as mx
from chess_engine.board import Board, Color, PieceType


def encode_board(board: Board) -> mx.array:
    """
    Encode the chess board as a 14-channel 8x8 array.
    Channels are:
    0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    6-11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    12: Current turn (1 for white, 0 for black)
    13: Castle rights (simplified - 1 where castling is possible)
    """
    # Initialize 14-channel board representation
    encoded = mx.zeros((14, 8, 8))

    # Map piece types to channel indices
    piece_to_channel = {
        (PieceType.PAWN, Color.WHITE): 0,
        (PieceType.KNIGHT, Color.WHITE): 1,
        (PieceType.BISHOP, Color.WHITE): 2,
        (PieceType.ROOK, Color.WHITE): 3,
        (PieceType.QUEEN, Color.WHITE): 4,
        (PieceType.KING, Color.WHITE): 5,
        (PieceType.PAWN, Color.BLACK): 6,
        (PieceType.KNIGHT, Color.BLACK): 7,
        (PieceType.BISHOP, Color.BLACK): 8,
        (PieceType.ROOK, Color.BLACK): 9,
        (PieceType.QUEEN, Color.BLACK): 10,
        (PieceType.KING, Color.BLACK): 11,
    }

    # Encode pieces
    for row in range(8):
        for col in range(8):
            piece = board.squares[row][col]
            if piece:
                channel = piece_to_channel.get((piece.piece_type, piece.color))
                if channel is not None:
                    encoded[channel, row, col] = 1

    # Encode current turn
    if board.current_turn == Color.WHITE:
        encoded[12] = mx.ones((8, 8))

    # Encode castling rights (simplified version)
    for row in [0, 7]:
        king = board.squares[row][4]
        if king and not king.has_moved:
            rook_kingside = board.squares[row][7]
            rook_queenside = board.squares[row][0]
            if rook_kingside and not rook_kingside.has_moved:
                encoded[13, row, 4:8] = 1
            if rook_queenside and not rook_queenside.has_moved:
                encoded[13, row, 0:5] = 1

    return encoded  # Shape: [14, 8, 8] in NCHW format


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
