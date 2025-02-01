import math
import time
import numpy as np
from typing import List, Tuple, Set, Dict
import mlx.core as mx
from chess_engine.bitboard import BitBoard
from config.model_config import ModelConfig
from functools import lru_cache


class Node:
    """Tree node storing state, prior probabilities, visit counts, and Q-values"""

    def __init__(self, board: BitBoard, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.children = {}
        # Use numpy for these operations
        self.visit_count = np.array(0)
        self.value_sum = np.array(0.0)
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return float(self.value_sum / self.visit_count)  # Convert back to Python float

    def select_child(self, c_puct):
        """Use numpy for UCB computation"""
        if not self.children:
            return None, None

        moves = list(self.children.keys())
        children = list(self.children.values())

        # Use numpy arrays for vectorized operations
        visit_counts = np.array([child.visit_count for child in children])
        values = np.array(
            [child.value_sum / max(1, vc) for child, vc in zip(children, visit_counts)]
        )
        priors = np.array([child.prior for child in children])

        # Vectorized UCB computation
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
        self.valid_moves_cache = {}
        self.position_cache = {}
        self.all_moves_cache = {}
        self.training = True
        self.training_prior_threshold = -1.0
        self.eval_prior_threshold = -1.0
        self.root_node = None  # Initialize root_node

        # Pre-compute move encoding table and buffers
        self._move_encoding_table = self._init_move_encoding_table()
        self.max_batch_size = 128
        self.policy_buffer = np.zeros((self.max_batch_size, 4096))
        self.value_buffer = np.zeros(self.max_batch_size)
        self.visit_counts_buffer = np.zeros(128)
        self.values_buffer = np.zeros(128)
        self.priors_buffer = np.zeros(128)

        # Pre-allocate buffers
        self.boards_buffer = np.zeros((128, 19, 8, 8), dtype=np.float32)
        self.moves_buffer = np.zeros((218, 2), dtype=np.int8)

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
        """Keep MLX only for policy/value conversion"""
        board = node.board
        policy_np = np.array(policy)

        # Get valid moves
        board_hash = node.board.get_hash()  # Use faster hashing
        if board_hash in self.position_cache:
            all_valid_moves = self.position_cache[board_hash]
        else:
            all_valid_moves = self._get_all_valid_moves(node.board)
            self.position_cache[board_hash] = all_valid_moves

        # Early exit if no moves
        total_moves = sum(len(moves) for moves in all_valid_moves.values())
        if total_moves == 0:
            return

        # Pre-allocate arrays for better memory efficiency
        child_boards = [None] * total_moves
        move_pairs = [None] * total_moves
        priors = np.empty(total_moves, dtype=np.float32)  # Single array allocation

        # Keep policy in MLX format, avoid numpy conversion
        idx = 0
        for from_pos, valid_moves in all_valid_moves.items():
            for to_pos in valid_moves:
                move_idx = self._move_encoding_table[(from_pos, to_pos)]
                # Access MLX array directly without conversion
                priors[idx] = float(policy_np[move_idx])

                # Reuse board objects when possible
                child_board = board.copy()
                child_board.make_move(from_pos, to_pos)

                child_boards[idx] = child_board
                move_pairs[idx] = (from_pos, to_pos)
                idx += 1

        # Create all nodes at once using dict comprehension
        node.children = {
            move: Node(board=board, parent=node, prior=float(prior))
            for move, board, prior in zip(move_pairs, child_boards, priors)
        }

        node.is_expanded = True
        node.value_sum = float(value)  # Convert MLX scalar to float once
        node.visit_count = 1

    def get_move(self, board: BitBoard):
        """Keep MLX only for model inference"""
        if self.debug:
            print("\nCurrent position:")
            print(board)

        self.model.eval()
        try:
            # Create root and do initial evaluation
            self.root_node = Node(board)  # Store root node as instance variable
            board_state = mx.array(board.state, dtype=mx.float32)[None, ...]
            policies, values = self.model(board_state)
            self._expand_node(self.root_node, policies[0], values[0].item())

            if not self.root_node.children:
                return None

            # Pre-allocate buffers once
            boards_buffer = np.zeros(
                (self.max_batch_size, *board.state.shape), dtype=np.float32
            )
            paths_buffer = [[] for _ in range(self.max_batch_size)]

            # Batch process simulations with early stopping check
            for sim_start in range(0, self.config.n_simulations, self.max_batch_size):
                batch_size = min(
                    self.max_batch_size, self.config.n_simulations - sim_start
                )

                self._batch_simulate(
                    self.root_node,
                    batch_size,
                    boards_buffer[:batch_size],
                    paths_buffer[:batch_size],
                )

                # Check if a move clearly dominates
                if self.should_stop(self.root_node):
                    if self.debug:
                        print(
                            f"Stopping simulations early at simulation {sim_start + batch_size} of {self.config.n_simulations}"
                        )
                    break

            # Return most visited move
            return max(self.root_node.children.items(), key=lambda x: x[1].visit_count)[
                0
            ]

        finally:
            self.model.train()

    def backup(self, search_path: List[Node], value: float):
        """Fully vectorized backup"""
        path_length = len(search_path)
        if path_length == 0:
            return

        # Pre-allocate value array
        values = np.full(path_length, value, dtype=np.float32)
        values[1::2] *= -1  # Flip values for opponent nodes

        # Update all nodes in one operation
        for node, val in zip(search_path, values):
            node.visit_count += 1
            node.value_sum += val

    def _get_valid_moves(
        self, board: BitBoard, pos: Tuple[int, int]
    ) -> Set[Tuple[int, int]]:
        """Get valid moves with caching"""
        board_hash = board.get_hash()
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
        board_hash = board.get_hash()
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

    def _get_moves_fast(
        self, board: BitBoard, moves_buffer: np.ndarray
    ) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
        """Fast move generation using pre-allocated buffer"""
        moves = {}
        pieces = board.get_all_pieces(board.get_current_turn())

        for pos, piece_type in pieces:
            valid_moves = board.get_valid_moves(pos)
            if valid_moves:
                moves[pos] = valid_moves

        return moves

    def _batch_simulate(
        self,
        root: Node,
        batch_size: int,
        boards_buffer: np.ndarray,
        paths_buffer: List[List[Node]],
    ):
        """Batch process multiple MCTS simulations"""
        # Selection phase - batch process
        nodes_to_expand = []
        for i in range(batch_size):
            node = root
            path = []
            path.append(node)

            # Tree traversal
            while node.is_expanded and node.children:
                move, child = node.select_child(self.config.c_puct)
                if not move:
                    break
                node = child
                path.append(node)

            if not node.board.is_game_over():
                boards_buffer[len(nodes_to_expand)] = node.board.state
                nodes_to_expand.append(node)
            paths_buffer[i] = path

        # Batch evaluate positions if any
        if nodes_to_expand:
            # Convert boards for model inference
            n_boards = len(nodes_to_expand)
            board_batch = mx.array(boards_buffer[:n_boards], dtype=mx.float32)
            policies, values = self.model(board_batch)

            # Expand nodes in batch
            for node, policy, value in zip(nodes_to_expand, policies, values):
                self._expand_node(node, policy, value.item())

        # Backup phase
        for path in paths_buffer[:batch_size]:
            if not path:  # Skip empty paths
                continue
            value = (
                path[-1].board.get_game_result()
                if path[-1].board.is_game_over()
                else path[-1].value()
            )
            self.backup(path, value)

    def should_stop(self, root: Node) -> bool:
        """
        Check if we can stop further simulations early.
        For example, if the best child visit count is sufficiently high compared
        to the others, we can return True.
        """
        if not root.children:
            return False
        # Gather visit counts from all children of the root
        visit_counts = [child.visit_count for child in root.children.values()]
        best = max(visit_counts)
        if len(visit_counts) > 1:
            # Get second best, note that the list is unsorted so sort it
            sorted_counts = sorted(visit_counts, reverse=True)
            second_best = sorted_counts[1]
        else:
            second_best = 0

        # If best has reached a minimal threshold and is significantly ahead of second best, stop.
        if best >= 20 and best > second_best * 1.5:
            if self.debug:
                print(
                    f"Early stop triggered: best = {best}, second best = {second_best}"
                )
            return True
        return False


class BitBoard:
    @lru_cache(maxsize=1024)
    def is_game_over(self) -> bool:
        """Cached game over check"""
        current_turn = self.get_current_turn()
        return (
            self.is_checkmate(current_turn)
            or self.is_stalemate(current_turn)
            or self.is_draw()
        )

    @lru_cache(maxsize=1024)
    def is_in_check(self, color: int) -> bool:
        """Cached check detection"""
        king_pos = self.get_king_position(color)
        return self._is_square_attacked_vectorized(king_pos, 1 - color)
