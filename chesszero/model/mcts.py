import math

from tqdm import tqdm
from chess_engine.board import Board, Color
from config.model_config import ModelConfig
from utils.board_utils import encode_board
from typing import List, Tuple
import mlx.core as mx


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
            return 0.0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct):
        """Select child with highest UCB score"""
        if not self.children:
            return None, None

        best_score = float("-inf")
        best_move = None
        best_child = None

        # Calculate these once instead of for each child
        total_visits = max(1, self.visit_count)
        sqrt_total_visits = math.sqrt(total_visits)

        for move, child in self.children.items():
            q_value = -child.value() if child.visit_count > 0 else 0
            u_value = c_puct * child.prior * sqrt_total_visits / (1 + child.visit_count)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child


class MCTS:
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

            # Run MCTS simulations in batches
            batch_size = min(
                32, self.config.n_simulations
            )  # Adjust batch size as needed
            for _ in range(0, self.config.n_simulations, batch_size):
                search_paths = []
                nodes_to_expand = []

                # Collect batch_size paths
                for _ in range(batch_size):
                    node = root
                    search_path = [node]

                    # Selection
                    while node.is_expanded:
                        move, child = node.select_child(self.config.c_puct)
                        if not move:  # No valid moves
                            break
                        node = child
                        search_path.append(node)

                    if not node.board.is_game_over():
                        nodes_to_expand.append(node)
                    search_paths.append((search_path, node))

                # Batch process expansions
                if nodes_to_expand:
                    # Prepare batch of board states
                    encoded_boards = mx.stack(
                        [
                            encode_board(node.board)[None, ...]
                            for node in nodes_to_expand
                        ]
                    )

                    # Get model predictions in batch
                    policies, values = self.model(encoded_boards)

                    # Process each node with its predictions
                    for node, policy, value in zip(nodes_to_expand, policies, values):
                        self._expand_node(node, policy[0], value[0])

                # Backup
                for search_path, end_node in search_paths:
                    value = (
                        end_node.board.get_game_result()
                        if end_node.board.is_game_over()
                        else end_node.value()
                    )
                    self.backup(search_path, value)

            # Select most visited move after search
            return max(root.children.items(), key=lambda x: x[1].visit_count)[0]

        finally:
            self.model.train()

    def expand_and_evaluate(self, node: Node):
        """Expand node and return value estimate"""
        # Get model predictions
        encoded_board = encode_board(node.board)
        policy, value = self.model(encoded_board[None, ...])

        self._expand_node(node, policy[0], value[0])
        return value[0]

    def backup(self, search_path: List[Node], value: float):
        """Update statistics of all nodes in search path"""
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Value flips for opponent

    def encode_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
        """Encode a move into a policy index"""
        from_idx = from_pos[0] * 8 + from_pos[1]
        to_idx = to_pos[0] * 8 + to_pos[1]
        return from_idx * 64 + to_idx

    def _expand_node(self, node: Node, policy, value):
        """Helper method to expand a single node with given policy and value"""
        pieces = (
            node.board.white_pieces
            if node.board.current_turn == Color.WHITE
            else node.board.black_pieces
        )

        for piece, pos in pieces:
            valid_moves = node.board.get_valid_moves(pos)
            for to_pos in valid_moves:
                move_idx = self.encode_move(pos, to_pos)
                prior = policy[move_idx]

                child_board = node.board.copy()
                child_board.move_piece(pos, to_pos)
                node.children[(pos, to_pos)] = Node(
                    board=child_board, parent=node, prior=prior
                )

        node.is_expanded = True
        node.value_sum = value
        node.visit_count = 1
