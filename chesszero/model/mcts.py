import math
import numpy as np
from chess_engine.board import Board, Color
from config.model_config import ModelConfig
from utils.board_utils import encode_board, decode_policy


class Node:
    """Tree node storing state, prior probabilities, visit counts, and Q-values"""

    def __init__(self, board: Board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.children = {}  # (from_pos, to_pos) -> Node
        self.visit_count = 0
        self.value_sum = 0
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct):
        """Select child with highest UCB score"""
        best_score = float("-inf")
        best_move = None

        for move, child in self.children.items():
            # UCB score = Q + U
            # Q = child value
            # U = prior * sqrt(parent visits) / (1 + child visits)
            q_value = (
                -child.value()
            )  # Negative because value is from opponent's perspective
            u_value = (
                c_puct
                * child.prior
                * math.sqrt(self.visit_count)
                / (1 + child.visit_count)
            )
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_move = move

        return best_move, self.children[best_move]


class MCTS:
    """
    MCTS implementation with:
    - Selection using PUCT algorithm
    - Expansion using network predictions
    - Backup using value estimates
    - Tree reuse between moves
    """

    def __init__(self, model, config: ModelConfig):
        self.model = model
        self.config = config

    def get_move(self, board: Board):
        """Get the best move for the current position after running MCTS"""
        # Run simulations
        root = Node(board)
        for _ in range(self.config.n_simulations):
            node = root
            value = self.expand_and_evaluate(node)

        # Select move based on visit counts
        move_probs = np.zeros(self.config.policy_output_dim)
        for child in root.children:
            move_probs[child.move_index] = child.visit_count
        move_probs = move_probs / np.sum(move_probs)

        # Select move (during training, sample from distribution)
        move_idx = np.random.choice(len(move_probs), p=move_probs)
        selected_move = self.decode_move(move_idx, board)

        # Print move info
        print(f"\nPosition evaluation: {root.value():.3f}")
        print(f"Selected move: {selected_move}")
        print(f"Move probability: {move_probs[move_idx]:.3f}")

        return selected_move

    def expand_and_evaluate(self, node: Node):
        """Expand node and return value estimate"""
        # Get model predictions
        encoded_board = encode_board(node.board)
        policy, value = self.model(encoded_board[None, ...])

        # Convert policy to moves
        moves = decode_policy(policy[0])

        # Add children for all legal moves
        pieces = (
            node.board.white_pieces
            if node.board.current_turn == Color.WHITE
            else node.board.black_pieces
        )

        for piece, from_pos in pieces:
            valid_moves = node.board.get_valid_moves(from_pos)
            for to_pos in valid_moves:
                # Find policy probability for this move
                move_idx = self.encode_move(from_pos, to_pos)
                prior = policy[0, move_idx] if move_idx < len(policy[0]) else 0.0

                # Create child node
                child_board = node.board.copy()
                child_board.move_piece(from_pos, to_pos)
                node.children[(from_pos, to_pos)] = Node(child_board, node, prior)

        node.is_expanded = True
        return value[0]

    def backup(self, search_path, value):
        """Update statistics of all nodes in search path"""
        for node in search_path:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Value flips for opponent

    def encode_move(self, from_pos, to_pos):
        """Convert move to policy index"""
        # This is a simplified version - we'll need to handle all move types properly
        from_square = from_pos[0] * 8 + from_pos[1]
        to_square = to_pos[0] * 8 + to_pos[1]
        return from_square * 64 + to_square
