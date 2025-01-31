import math
import time
import numpy as np
from typing import List, Tuple, Set, Dict
import mlx.core as mx
from chess_engine.bitboard import BitBoard
from config.model_config import ModelConfig


class Node:
    """Tree node storing state, prior probabilities, visit counts, and Q-values"""

    def __init__(self, board: BitBoard, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.children = {}  # (from_pos, to_pos) -> Node
        self.visit_count = np.array(0)  # Store as numpy scalar
        self.value_sum = np.array(0.0)  # Store as numpy scalar
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return float(self.value_sum / self.visit_count)  # Convert back to Python float

    def select_child(self, c_puct):
        """Select child with highest UCB score"""
        if not self.children:
            return None, None

        moves = list(self.children.keys())
        children = list(self.children.values())

        # Convert to numpy arrays
        visit_counts = np.array([child.visit_count for child in children])
        values = np.array(
            [child.value_sum / max(1, vc) for child, vc in zip(children, visit_counts)]
        )
        priors = np.array([child.prior for child in children])

        # Vectorized UCB computation on CPU
        sqrt_total_visits = math.sqrt(max(1, self.visit_count))
        q_values = np.where(visit_counts > 0, -values, 0)
        ucb_scores = q_values + c_puct * sqrt_total_visits * priors / (1 + visit_counts)

        best_idx = int(np.argmax(ucb_scores))
        return moves[best_idx], children[best_idx]


class MCTS:
    def __init__(self, model, config: ModelConfig):
        self.model = model
        self.config = config
        self.debug = True
        self.valid_moves_cache = {}  # (board_hash, position) -> valid moves
        self.all_moves_cache = {}  # board_hash -> all valid moves
        self.training = True
        self.training_prior_threshold = -1.0
        self.eval_prior_threshold = -1.0

        # Pre-compute move encoding table and buffers
        self._move_encoding_table = self._init_move_encoding_table()
        self.max_batch_size = 128
        self.policy_buffer = np.zeros((self.max_batch_size, 4096))
        self.value_buffer = np.zeros(self.max_batch_size)

    def _init_move_encoding_table(
        self,
    ) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], int]:
        """Pre-compute move index for every possible (from_pos, to_pos) combination"""
        table = {}
        for from_row in range(8):
            for from_col in range(8):
                for to_row in range(8):
                    for to_col in range(8):
                        from_pos = (from_row, from_col)
                        to_pos = (to_row, to_col)
                        from_idx = from_row * 8 + from_col
                        to_idx = to_row * 8 + to_col
                        move_idx = from_idx * 64 + to_idx
                        table[(from_pos, to_pos)] = move_idx
        return table

    def encode_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
        """Fast move encoding using lookup table"""
        return self._move_encoding_table[(from_pos, to_pos)]

    def _expand_node(self, node: Node, policy, value):
        """Optimized node expansion"""
        board = node.board
        pieces = board.get_all_pieces(board.get_current_turn())
        policy_np = np.array(policy)

        # Get all valid moves at once
        all_valid_moves = {}
        for pos, piece_type in pieces:
            valid_moves = board.get_valid_moves(pos)
            if valid_moves:
                all_valid_moves[pos] = valid_moves

        # Batch create child boards
        child_boards = []
        move_pairs = []
        priors = []

        # Collect all moves first
        for from_pos, valid_moves in all_valid_moves.items():
            for to_pos in valid_moves:
                move_idx = self._move_encoding_table[(from_pos, to_pos)]
                prior = policy_np[move_idx]

                child_board = board.copy()
                child_board.make_move(from_pos, to_pos)

                child_boards.append(child_board)
                move_pairs.append((from_pos, to_pos))
                priors.append(prior)

        # Create all nodes at once
        node.children = {
            move: Node(board=child_board, parent=node, prior=prior)
            for child_board, move, prior in zip(child_boards, move_pairs, priors)
        }

        node.is_expanded = True
        node.value_sum = value
        node.visit_count = 1

    def get_move(self, board: BitBoard):
        """Get the best move for the current position"""
        print("\nCurrent position:")
        print(board)

        self.model.eval()
        try:
            root = Node(board)

            # Initial expansion
            board_state = mx.array(board.state, dtype=mx.float32)[None, ...]
            policies, values = self.model(board_state)
            self._expand_node(root, policies[0], values[0].item())

            if not root.children:
                return None

            # Process simulations in batches
            batch_size = self.config.n_simulations
            chunk_size = self.max_batch_size

            boards_buffer = []
            nodes_buffer = []
            paths_buffer = []

            for sim_start in range(0, batch_size, chunk_size):
                chunk_end = min(sim_start + chunk_size, batch_size)

                # Selection phase
                for _ in range(chunk_end - sim_start):
                    node = root
                    path = [node]

                    while node.is_expanded and node.children:
                        move, child = node.select_child(self.config.c_puct)
                        if not move:
                            break
                        node = child
                        path.append(node)

                    if not node.board.is_game_over():
                        boards_buffer.append(node.board.state)
                        nodes_buffer.append(node)
                    paths_buffer.append(path)

                # Batch evaluate positions
                if boards_buffer:
                    board_batch = mx.array(np.stack(boards_buffer), dtype=mx.float32)
                    policies, values = self.model(board_batch)

                    for node, policy, value in zip(nodes_buffer, policies, values):
                        self._expand_node(node, policy, value.item())

                    boards_buffer.clear()
                    nodes_buffer.clear()

                # Backup phase
                for path in paths_buffer:
                    value = (
                        path[-1].board.get_game_result()
                        if path[-1].board.is_game_over()
                        else path[-1].value()
                    )
                    self.backup(path, value)
                paths_buffer.clear()

            # Return most visited move
            return max(root.children.items(), key=lambda x: x[1].visit_count)[0]

        finally:
            self.model.train()

    def backup(self, search_path: List[Node], value: float):
        """Vectorized backup using numpy arrays"""
        path_length = len(search_path)
        values = np.full(path_length, value)
        values[1::2] *= -1  # Flip values for opponent nodes

        # Update all nodes at once
        for node, val in zip(search_path, values):
            node.visit_count += 1
            node.value_sum += val

    def _get_valid_moves(
        self, board: BitBoard, pos: Tuple[int, int]
    ) -> Set[Tuple[int, int]]:
        """Get valid moves with caching"""
        board_hash = hash(str(board.state))
        cache_key = (board_hash, pos)

        if cache_key in self.valid_moves_cache:
            return self.valid_moves_cache[cache_key]

        valid_moves = board.get_valid_moves(pos)
        self.valid_moves_cache[cache_key] = valid_moves
        return valid_moves

    def _get_all_valid_moves(
        self, board: BitBoard
    ) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
        """Get all valid moves for current player with caching"""
        board_hash = hash(str(board.state))
        if board_hash in self.all_moves_cache:
            return self.all_moves_cache[board_hash]

        moves = {}
        pieces = board.get_all_pieces(board.get_current_turn())
        for pos, _ in pieces:
            valid_moves = board.get_valid_moves(pos)
            if valid_moves:
                moves[pos] = valid_moves

        self.all_moves_cache[board_hash] = moves
        return moves
