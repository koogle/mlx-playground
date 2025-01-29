import math
import numpy as np
from tqdm import tqdm
from chess_engine.board import Board, Color
from config.model_config import ModelConfig
from utils.board_utils import encode_board, decode_policy
from typing import List, Tuple


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
        """Get the average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct):
        """Select child with highest UCB score"""
        if not self.children:  # No children available
            return None, None

        best_score = float("-inf")
        best_move = None
        total_visits = max(1, self.visit_count)  # Ensure at least 1 visit

        for move, child in self.children.items():
            # Negative because value is from opponent's perspective
            q_value = float(-child.value() if child.visit_count > 0 else 0)

            # Calculate exploration bonus
            u_value = float(
                c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visit_count)
            )

            # Combine Q-value and exploration bonus
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_move = move

        if best_move is None:  # No valid moves found
            return None, None

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
        # Set model to eval mode during search
        self.model.eval()
        try:
            root = Node(board)
            value = self.expand_and_evaluate(root)
            if not root.children:
                return None

            # Run simulations with early stopping
            n_sims = min(100, self.config.n_simulations)  # Reduced simulation count
            for idx in range(n_sims):
                # Selection with depth limit
                node = root
                search_path = [node]
                depth = 0
                max_depth = 15  # Further reduced max depth

                # Fast selection loop
                while node.is_expanded and depth < max_depth:
                    # Get best child using PUCT
                    best_score = float("-inf")
                    best_move = None
                    best_child = None

                    # Single pass through children
                    for move, child in node.children.items():
                        q_value = -child.value() if child.visit_count > 0 else 0
                        u_value = (
                            self.config.c_puct
                            * child.prior
                            * (math.sqrt(node.visit_count) / (1 + child.visit_count))
                        )
                        score = q_value + u_value

                        if score > best_score:
                            best_score = score
                            best_move = move
                            best_child = child

                    if not best_child:
                        break

                    node = best_child
                    search_path.append(node)
                    depth += 1

                # Expansion and evaluation
                if not node.is_expanded and depth < max_depth:
                    value = self.expand_and_evaluate(node)
                    if not node.children:
                        value = -1

                # Backup
                self.backup(search_path, value)

                # Early stopping - check less frequently
                if idx > 20 and idx % 5 == 0:
                    best_visits = max(
                        child.visit_count for child in root.children.values()
                    )
                    total_visits = sum(
                        child.visit_count for child in root.children.values()
                    )

                    if total_visits > 0 and best_visits / total_visits > 0.75:
                        break

            # Return most visited move
            return max(root.children.items(), key=lambda x: x[1].visit_count)[0]

        finally:
            self.model.train()

    def expand_and_evaluate(self, node: Node):
        """Expand node and return value estimate"""
        try:
            # Get model predictions
            encoded_board = encode_board(node.board)
            policy, value = self.model(encoded_board[None, ...])

            # Add children for all legal moves
            pieces = (
                node.board.white_pieces
                if node.board.current_turn == Color.WHITE
                else node.board.black_pieces
            )

            for piece, from_pos in pieces:
                valid_moves = node.board.get_valid_moves(from_pos)
                for to_pos in valid_moves:
                    # Get policy probability for this move
                    move_idx = self.encode_move(from_pos, to_pos)
                    prior = policy[0, move_idx] if move_idx < len(policy[0]) else 0.0

                    # Create child node
                    child_board = node.board.copy()
                    child_board.move_piece(from_pos, to_pos)
                    node.children[(from_pos, to_pos)] = Node(
                        board=child_board, parent=node, prior=prior
                    )

            node.is_expanded = True
            return value[0]

        except Exception as e:
            print(f"Error in expand_and_evaluate: {e}")
            # Return neutral evaluation on error
            return 0.0

    def backup(self, search_path: List[Node], value: float):
        """Update statistics of all nodes in search path"""
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Value flips for opponent

    def encode_move(self, from_pos, to_pos):
        """Convert move to policy index"""
        # This is a simplified version - we'll need to handle all move types properly
        from_square = from_pos[0] * 8 + from_pos[1]
        to_square = to_pos[0] * 8 + to_pos[1]
        return from_square * 64 + to_square
