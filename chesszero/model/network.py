class ChessNet:
    """
    Combined policy and value network following AlphaZero architecture:
    - Input: 8x8x119 board representation (current state + history)
    - Body: ResNet with residual blocks
    - Policy head: Outputs move probabilities (8x8x73 for all possible moves)
    - Value head: Outputs scalar position evaluation
    """
