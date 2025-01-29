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
        if not self.children:
            return None, None

        best_score = float("-inf")
        best_move = None
        best_child = None

        total_visits = max(1, self.visit_count)

        for move, child in self.children.items():
            q_value = -child.value() if child.visit_count > 0 else 0

            if math.isnan(child.prior):
                raise ValueError(f"ERROR: Prior is NaN for move {move}")
            if math.isnan(child.visit_count):
                raise ValueError(f"ERROR: Visit count is NaN for move {move}")

            u_value = (
                c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visit_count)
            )
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child


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
        self.model.eval()
        try:
            root = Node(board)

            # First expand root node
            value = self.expand_and_evaluate(root)
            if not root.children:
                return None

            # Run MCTS simulations
            for _ in range(self.config.n_simulations):
                node = root
                search_path = [node]

                # Selection
                while node.is_expanded:
                    move, child = node.select_child(self.config.c_puct)
                    if not move:  # No valid moves
                        break
                    node = child
                    search_path.append(node)

                # Expansion and evaluation
                if node.board.is_game_over():
                    value = node.board.get_game_result()
                else:
                    value = self.expand_and_evaluate(node)

                # Backup
                self.backup(search_path, value)

            # Print search statistics
            print("\nSearch results:")
            moves_with_stats = []
            for move, child in root.children.items():
                moves_with_stats.append(
                    (move, child.visit_count, child.value(), child.prior)
                )

            # Sort by visit count
            moves_with_stats.sort(key=lambda x: x[1], reverse=True)

            # Print top moves
            for move, visits, value, prior in moves_with_stats[:10]:  # Show top 10
                print(
                    f"Move {move:15} Visits: {visits:4d} Value: {value:7.3f} Prior: {prior:7.3f}"
                )

            # Select most visited move after search
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

            for piece, pos in pieces:
                valid_moves = node.board.get_valid_moves(pos)
                for to_pos in valid_moves:
                    move_idx = self.encode_move(pos, to_pos)
                    prior = policy[0, move_idx]

                    child_board = node.board.copy()
                    child_board.move_piece(pos, to_pos)
                    node.children[(pos, to_pos)] = Node(
                        board=child_board, parent=node, prior=prior
                    )

            node.is_expanded = True
            return value[0]

        except Exception as e:
            print(f"Error in expand_and_evaluate: {e}")
            import traceback

            print(traceback.format_exc())
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
