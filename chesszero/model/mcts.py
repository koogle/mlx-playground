class Node:
    """Tree node storing state, prior probabilities, visit counts, and Q-values"""


class MCTS:
    """
    MCTS implementation with:
    - Selection using PUCT algorithm
    - Expansion using network predictions
    - Backup using value estimates
    - Tree reuse between moves
    """
