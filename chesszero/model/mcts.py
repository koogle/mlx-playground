import math
import numpy as np
from typing import List, Tuple, Set, Dict
import mlx.core as mx
from chess_engine.bitboard import BitBoard
from config.model_config import ModelConfig
import random


class Node:
    """Tree node storing state, prior probabilities, visit counts, and Q-values"""

    def __init__(self, board: BitBoard, parent=None, prior=0.0):
        self.visit_count = 0
        self.value_sum = 0.0
        self.board = board
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count  # already a Python float

    def select_child(self, c_puct):
        """Optimized child selection"""
        if not self.children:
            return None, None

        # Use list comprehension instead of numpy for small arrays
        visit_counts = [child.visit_count for child in self.children.values()]
        total_visits = sum(visit_counts)
        sqrt_total = math.sqrt(total_visits)

        # Calculate UCB scores directly
        best_score = float("-inf")
        best_move = None
        best_child = None

        for move, child in self.children.items():
            if child.visit_count == 0:
                q_value = 0
            else:
                q_value = (
                    -child.value()
                )  # Negative because we want to maximize opponent's loss

            # UCB formula
            exploration = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            ucb_score = q_value + exploration

            if child.parent is None:  # Root node
                ucb_score = (
                    0.75 * ucb_score + 0.25 * random.random()
                )  # Simplified noise

            if ucb_score > best_score:
                best_score = ucb_score
                best_move = move
                best_child = child

        return best_move, best_child


class MCTS:
    def __init__(self, model, config: ModelConfig):
        self.model = model
        self.config = config
        self.debug = False
        self.cache_max_size = 50000  # Adjust this based on memory constraints
        self.valid_moves_cache = {}
        self.position_cache = {}
        self.all_moves_cache = {}
        self.training = True
        self.training_prior_threshold = -1.0
        self.eval_prior_threshold = -1.0
        self.root_node = None  # Initialize root_node
        self.transposition_table = {}  # Store equivalent positions

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

        self._value_cache = {}  # Cache for evaluated positions
        self._policy_cache = {}  # Cache for policy evaluations
        self._tree_cache = {}  # Cache for subtrees

    def clear_all_caches(self):
        """Clear all caches between games"""
        self.valid_moves_cache.clear()
        self.position_cache.clear()
        self.all_moves_cache.clear()
        self.transposition_table.clear()
        self._value_cache.clear()
        self._policy_cache.clear()
        self._tree_cache.clear()
        self.root_node = None
        # Reset counters
        self.total_nodes_visited = 0
        self.moves_made = 0

    def _init_move_encoding_table(
        self,
    ) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], int]:
        """Pre-compute move index for every possible (from_pos, to_pos) combination, including pawn promotions"""
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

                        # Add promotion moves
                        if (from_row == 6 and to_row == 7) or (  # White pawn promoting
                            from_row == 1 and to_row == 0
                        ):  # Black pawn promoting
                            for promotion_piece in range(
                                1, 5
                            ):  # Knight, Bishop, Rook, Queen
                                table[(from_pos, (to_row, to_col, promotion_piece))] = (
                                    move_idx
                                )

        return table

    def encode_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
        """Fast move encoding using lookup table"""
        return self._move_encoding_table[(from_pos, to_pos)]

    def _evaluate_position(self, board_state: np.ndarray):
        """Cache model evaluations"""
        state_hash = hash(board_state.tobytes())
        if state_hash in self._value_cache:
            return self._policy_cache[state_hash], self._value_cache[state_hash]

        board_state = mx.array(board_state, dtype=mx.float32)[None, ...]
        policies, values = self.model(board_state)

        self._policy_cache[state_hash] = policies[0]
        self._value_cache[state_hash] = values[0].item()
        return policies[0], values[0].item()

    def _expand_node(self, node: Node, policy, value):
        """Cache common subtrees and prevent cycles"""
        board_hash = node.board.get_hash()

        # Check if this position has occurred in the current search path
        current = node
        path_positions = {board_hash}
        while current.parent:
            parent_hash = current.parent.board.get_hash()
            if parent_hash in path_positions:
                # Position repeats - treat as a draw
                node.is_expanded = True
                node.value_sum = 0.0  # Draw value
                node.visit_count = 1
                return
            path_positions.add(parent_hash)
            current = current.parent

        # Rest of existing expansion code...
        if board_hash in self._tree_cache:
            node.children = self._tree_cache[board_hash]
            node.is_expanded = True
            node.value_sum = value
            node.visit_count = 1
            return

        board = node.board
        policy_np = np.array(policy)

        # Get valid moves using cached lookup
        board_hash = node.board.get_hash()
        all_valid_moves = self.position_cache.get(board_hash)
        if all_valid_moves is None:
            all_valid_moves = self._get_all_valid_moves(node.board)
            self.position_cache[board_hash] = all_valid_moves

        # Early exit if no moves
        if not all_valid_moves:
            return

        # Calculate priors more efficiently
        total_prior = 0.0
        children = {}

        for from_pos, valid_moves in all_valid_moves.items():
            for to_pos in valid_moves:
                move_idx = self._move_encoding_table[(from_pos, to_pos)]
                prior = max(float(policy_np[move_idx]), 1e-8)
                total_prior += prior

                # Create child board and node
                child_board = board.copy()
                child_board.make_move(from_pos, to_pos)
                children[(from_pos, to_pos)] = Node(
                    board=child_board, parent=node, prior=prior
                )

        # Normalize priors in one pass
        if total_prior > 0:
            for child in children.values():
                child.prior /= total_prior

        node.children = children
        node.is_expanded = True
        node.value_sum = float(value)
        node.visit_count = 1

        # Cache the subtree
        if len(node.children) > 0:
            self._tree_cache[board_hash] = node.children

    def _get_transposition_key(self, board: BitBoard) -> str:
        """Get unique key for equivalent positions"""
        return board.get_hash()

    def get_move(self, board: BitBoard):
        """Reuse calculations for transpositions"""
        # Don't switch to eval mode during training
        if not self.training:
            self.model.eval()

        try:
            # Clean caches periodically
            if len(self.valid_moves_cache) > self.cache_max_size:
                self._prune_caches()

            key = self._get_transposition_key(board)
            if key in self.transposition_table:
                cached_node = self.transposition_table[key]
                if cached_node.visit_count > self.config.n_simulations // 2:
                    return max(
                        cached_node.children.items(), key=lambda x: x[1].visit_count
                    )[0]

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
            # Restore training mode if we were training
            if self.training:
                self.model.train()

    def backup(self, search_path: List[Node], value: float):
        """Fully vectorized backup"""
        # Using native int/float updates
        # Precompute the values for each node along the path:
        values_along_path = [
            value if idx % 2 == 0 else -value for idx in range(len(search_path))
        ]
        for node, v in zip(search_path, values_along_path):
            node.visit_count += 1
            node.value_sum += v

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
        """Optimized batch simulation with dynamic batch sizing"""
        # Adjust batch size based on tree depth
        avg_path_length = self._estimate_path_length(root)
        adjusted_batch_size = min(
            batch_size, max(1, self.max_batch_size // avg_path_length)
        )

        nodes_to_expand = []
        board_states = []

        # Selection phase with early cutoff
        for i in range(adjusted_batch_size):
            node = root
            path = [node]
            depth = 0

            while node.is_expanded and node.children:
                move, child = node.select_child(self.config.c_puct)
                if not move:
                    break
                node = child
                path.append(node)
                depth += 1

                # Early cutoff for very deep paths
                if depth > 50:  # Avoid extremely deep searches
                    break

            if not node.board.is_game_over():
                board_states.append(node.board.state)
                nodes_to_expand.append(node)
            paths_buffer[i] = path

        # Batch evaluate positions if any
        if nodes_to_expand:
            # Convert boards for model inference - do it once
            board_batch = mx.array(np.stack(board_states), dtype=mx.float32)
            policies, values = self.model(board_batch)

            # Expand nodes in batch
            for node, policy, value in zip(nodes_to_expand, policies, values):
                self._expand_node(node, policy, value.item())

        # Backup phase - use native Python types for speed
        for path in paths_buffer[:adjusted_batch_size]:
            if not path:  # Skip empty paths
                continue
            value = (
                path[-1].board.get_game_result()
                if path[-1].board.is_game_over()
                else path[-1].value()
            )
            self.backup(path, value)

    def should_stop(self, root: Node) -> bool:
        """Determine if we can stop simulations early based on multiple criteria"""
        if not root.children:
            return False

        # Get visit counts and values
        moves_data = [
            (move, child.visit_count, child.value())
            for move, child in root.children.items()
        ]

        if len(moves_data) <= 1:
            return False

        # Sort by visit count
        moves_data.sort(key=lambda x: (x[1], x[2]), reverse=True)
        best_move, best_visits, best_value = moves_data[0]
        second_best = moves_data[1]

        # Early stopping criteria:

        # 1. Minimum visit threshold
        if best_visits < 20:
            return False

        # 2. Clear statistical dominance
        visit_ratio = best_visits / (second_best[1] + 1e-8)
        if visit_ratio > 3.0:  # Best move has 3x more visits
            return True

        # 3. Value convergence
        if best_visits > 50:
            value_diff = abs(best_value - second_best[2])
            if value_diff > 0.8:  # Clear value advantage
                return True

        # 4. Move quality threshold
        total_visits = sum(x[1] for x in moves_data)
        visit_percentage = best_visits / total_visits
        if visit_percentage > 0.8 and best_visits > 30:  # One move has 80% of visits
            return True

        # 5. Time efficiency for obvious moves
        if len(moves_data) > 2:
            rest_visits = sum(x[1] for x in moves_data[2:])
            if best_visits > rest_visits * 4:  # Best move dominates all others combined
                return True

        return False

    def _estimate_path_length(self, node: Node, sample_size: int = 5) -> int:
        """Estimate average path length in the tree"""
        if not node.children:
            return 1

        total_length = 0
        for _ in range(sample_size):
            current = node
            length = 1
            while current.children and length < 50:
                move, child = current.select_child(self.config.c_puct)
                if not move or not child:
                    break
                current = child
                length += 1
            total_length += length

        return max(1, total_length // sample_size)

    def _prune_caches(self):
        """More aggressive cache pruning"""
        # Keep only 75% of max size to prevent frequent pruning
        target_size = int(self.cache_max_size * 0.75)

        # Prune all caches
        self.valid_moves_cache = dict(
            list(self.valid_moves_cache.items())[-target_size:]
        )
        self.position_cache = dict(list(self.position_cache.items())[-target_size:])
        self.all_moves_cache = dict(list(self.all_moves_cache.items())[-target_size:])
        self._value_cache = dict(list(self._value_cache.items())[-target_size:])
        self._policy_cache = dict(list(self._policy_cache.items())[-target_size:])
        self._tree_cache = dict(list(self._tree_cache.items())[-target_size:])

        # Clear transposition table completely since it's less critical
        self.transposition_table.clear()

        if self.debug:
            print(f"Pruned caches to {target_size} entries")


class BitBoard:
    def is_game_over(self) -> bool:
        """Cached game over check"""
        current_turn = self.get_current_turn()
        return (
            self.is_checkmate(current_turn)
            or self.is_stalemate(current_turn)
            or self.is_draw()
        )

    def is_in_check(self, color: int) -> bool:
        """Cached check detection"""
        king_pos = self.get_king_position(color)
        return self._is_square_attacked_vectorized(king_pos, 1 - color)
