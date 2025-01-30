import math
from random import randint

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
        self.debug = False
        self.board_cache = {}
        # Add training flag and different thresholds
        self.training = True
        self.training_prior_threshold = 0.0  # No pruning during training
        self.eval_prior_threshold = (
            0.00  # Prune low probability moves during evaluation
        )

    def get_move(self, board: Board):
        """Get the best move for the current position after running MCTS"""
        self.model.eval()
        try:
            root = Node(board)
            # Fix array handling in expand_and_evaluate
            encoded_board = encode_board(board)[None, ...]
            policies, values = self.model(encoded_board)
            policy = mx.array(policies[0])  # Convert to MLX array
            value = values[0].item()  # Get scalar value

            self._expand_node(root, policy, value)

            if not root.children:
                print("\nWarning: Root node has no children after expansion!")
                print(f"Board state:\n{board}")
                print(f"Root value: {value}")
                valid_moves = []
                pieces = (
                    board.white_pieces
                    if board.current_turn == Color.WHITE
                    else board.black_pieces
                )
                for piece, pos in pieces:
                    moves = board.get_valid_moves(pos)
                    if moves:
                        valid_moves.extend([(pos, move) for move in moves])
                print(f"Valid moves: {valid_moves}")
                if self.debug:
                    print(f"Policy shape: {policy.shape}")
                    print(f"Policy range: [{mx.min(policy):.4f}, {mx.max(policy):.4f}]")
                return None if not valid_moves else valid_moves[0]

            # Pre-allocate arrays for batching
            batch_size = min(self.config.n_simulations, self.config.n_simulations)
            encoded_boards = []
            nodes_to_expand = []
            search_paths = []

            # Run simulations in batches
            encoded_boards.clear()
            nodes_to_expand.clear()
            search_paths.clear()

            # Collect batch_size paths
            for _ in range(batch_size):
                node = root
                path = [node]

                # Selection - avoid creating new boards until necessary
                while node.is_expanded and node.children:
                    move, child = node.select_child(self.config.c_puct)
                    if not move:
                        break
                    node = child
                    path.append(node)

                if not node.board.is_game_over():
                    # Cache board encoding
                    board_hash = hash(str(node.board))
                    if board_hash not in self.board_cache:
                        self.board_cache[board_hash] = encode_board(node.board)
                    encoded_boards.append(self.board_cache[board_hash])
                    nodes_to_expand.append(node)
                search_paths.append((path, node))

            # Batch evaluate positions
            if encoded_boards:
                encoded_batch = mx.stack(encoded_boards)
                policies, values = self.model(encoded_batch)

                if self.debug:
                    print("\nPolicy statistics:")
                    for i, policy in enumerate(policies):
                        print(
                            f"Policy {i}: min={mx.min(policy):.4f}, max={mx.max(policy):.4f}"
                        )

                # Expand nodes with predictions
                for node, policy, value in zip(nodes_to_expand, policies, values):
                    policy = mx.array(policy)  # Convert to MLX array
                    value = value.item()  # Get scalar value
                    self._expand_node(node, policy, value)

            # Backup
            for path, end_node in search_paths:
                value = (
                    end_node.board.get_game_result()
                    if end_node.board.is_game_over()
                    else end_node.value()
                )
                self.backup(path, value)

            # Select best move
            if not root.children:
                # Fallback to any valid move if no children
                valid_moves = []
                pieces = (
                    board.white_pieces
                    if board.current_turn == Color.WHITE
                    else board.black_pieces
                )
                for piece, pos in pieces:
                    moves = board.get_valid_moves(pos)
                    if moves:
                        valid_moves.extend([(pos, move) for move in moves])
                return valid_moves[0] if valid_moves else None

            return max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        finally:
            self.model.train()

    def expand_and_evaluate(self, node: Node):
        """Expand node and return value estimate"""
        # Get model predictions - add batch dimension but not extra dimension
        encoded_board = encode_board(node.board)[None, ...]
        policy, value = self.model(encoded_board)

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
        """Expand node with all valid moves"""
        board = node.board
        is_white = board.current_turn == Color.WHITE
        pieces = board.white_pieces if is_white else board.black_pieces

        if self.debug:
            print(f"\nExpanding node for {'White' if is_white else 'Black'}")
            print(f"Number of pieces: {len(pieces)}")
            print(f"Board state:\n{board}")

        # Create children for all valid moves
        children_created = 0
        for piece, pos in pieces:
            valid_moves = board.get_valid_moves(pos)
            if valid_moves:
                if self.debug:
                    print(f"Valid moves for {piece} at {pos}: {valid_moves}")

                for to_pos in valid_moves:
                    move_idx = self.encode_move(pos, to_pos)
                    prior = policy[move_idx]

                    # Create child board and make move
                    child_board = board.copy()
                    child_board.move_piece(pos, to_pos)
                    node.children[(pos, to_pos)] = Node(
                        board=child_board, parent=node, prior=prior
                    )
                    children_created += 1

        if self.debug:
            print(f"Created {children_created} child nodes")

        node.is_expanded = True
        node.value_sum = value
        node.visit_count = 1
