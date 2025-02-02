import numpy as np
from typing import List, Tuple, Set, Dict
import mlx.core as mx
from chess_engine.bitboard import BitBoard
from config.model_config import ModelConfig
import time


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
        """Optimized child selection with exploration"""
        if not self.children:
            return None, None

        moves = list(self.children.keys())
        children = list(self.children.values())

        # If no visits yet, sample based on prior probability
        if all(child.visit_count == 0 for child in children):
            priors = np.array([child.prior for child in children])

            # Add less Dirichlet noise at root
            if self.parent is None:
                alpha = 0.15  # Reduced from 0.3
                noise = np.random.dirichlet([alpha] * len(priors))
                priors = (
                    0.9 * priors + 0.1 * noise
                )  # More weight on priors (was 0.75/0.25)

            # Sample from prior distribution with higher temperature for first move
            priors = priors / np.sum(priors)
            selected_idx = np.argmax(priors)  # More greedy selection
            return moves[selected_idx], children[selected_idx]

        # Calculate UCB scores with lower temperature
        temperature = 0.5  # Reduced from 1.0 for less exploration
        visit_counts = np.array([child.visit_count for child in children])
        total_visits = np.sum(visit_counts)
        sqrt_total = np.sqrt(total_visits)

        q_values = np.array(
            [-child.value() if child.visit_count > 0 else 0.0 for child in children]
        )

        priors = np.array([child.prior for child in children])

        # UCB formula with reduced noise
        exploration = c_puct * priors * sqrt_total / (1 + visit_counts)
        ucb_scores = q_values + temperature * exploration

        # Reduced random noise
        ucb_scores += np.random.normal(
            0, 0.001, size=len(ucb_scores)
        )  # Reduced from 0.01ex

        selected_idx = np.argmax(ucb_scores)
        return moves[selected_idx], children[selected_idx]


class MCTS:
    def __init__(self, model, config: ModelConfig):
        self.model = model
        self.config = config
        self.debug = config.debug
        self.cache_max_size = 5000  # Adjust this based on memory constraints
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

        # Add debugging stats
        self.max_path_length = 0
        self.total_paths = 0
        self.path_lengths = []  # Track all path lengths
        self.current_game_moves = 0  # Track moves in current game

    def clear_all_caches(self):
        """Clear all caches between games and force garbage collection"""
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

        # Reset path statistics for new game
        self.max_path_length = 0
        self.path_lengths = []
        self.total_paths = 0
        self.current_game_moves = 0

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
        """Cache model evaluations with proper cleanup"""
        state_hash = hash(board_state.tobytes())
        if state_hash in self._value_cache:
            return self._policy_cache[state_hash], self._value_cache[state_hash]

        # Convert to MLX array
        board_state = mx.array(board_state, dtype=mx.float32)[None, ...]

        # Get model predictions
        policies, values = self.model(board_state)

        # Evaluate and convert to numpy/python types for caching
        mx.eval(policies, values)
        policy = policies[0]
        value = values[0].item()

        # Cache the results
        self._policy_cache[state_hash] = policy
        self._value_cache[state_hash] = value

        # Clean up intermediate tensors
        del board_state
        del policies
        del values

        return policy, value

    def _expand_node(self, node: Node, policy, value):
        """Expand node using board cache"""
        board_hash = node.board.get_hash()

        # Check for cycles
        current = node
        path_positions = {board_hash}
        while current.parent:
            parent_hash = current.parent.board.get_hash()
            if parent_hash in path_positions:
                node.is_expanded = True
                node.value_sum = 0.0
                node.visit_count = 1
                return
            path_positions.add(parent_hash)
            current = current.parent

        # Use cached subtree if available
        if board_hash in self._tree_cache:
            node.children = self._tree_cache[board_hash]
            node.is_expanded = True
            node.value_sum = value
            node.visit_count = 1
            return

        board = node.board
        policy_np = np.array(policy)
        all_valid_moves = self._get_all_valid_moves(board)

        if not all_valid_moves:
            return

        children = {}
        total_prior = 0.0

        for from_pos, valid_moves in all_valid_moves.items():
            for to_pos in valid_moves:
                move_idx = self._move_encoding_table[(from_pos, to_pos)]
                prior = max(float(policy_np[move_idx]), 1e-8)
                total_prior += prior

                child_board = board.copy()
                child_board.make_move(from_pos, to_pos)

                children[(from_pos, to_pos)] = Node(
                    board=child_board, parent=node, prior=prior
                )

        # Normalize priors
        if total_prior > 0:
            for child in children.values():
                child.prior /= total_prior

        node.children = children
        node.is_expanded = True
        node.value_sum = float(value)
        node.visit_count = 1

        if len(node.children) > 0:
            self._tree_cache[board_hash] = node.children

    def _get_transposition_key(self, board: BitBoard) -> str:
        """Get unique key for equivalent positions"""
        return board.get_hash()

    def get_move(self, board: BitBoard, temperature: float = 0.0):
        """Get best move using MCTS with optional temperature sampling"""
        self.current_game_moves += 1

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
                    return self._select_move_with_temperature(cached_node, temperature)

            # Create root and do initial evaluation
            self.root_node = Node(board)
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

            # Select move based on temperature
            move = self._select_move_with_temperature(self.root_node, temperature)
            self._cleanup_tree(self.root_node)
            return move

        finally:
            # Restore training mode if we were training
            if self.training:
                self.model.train()

    def _select_move_with_temperature(
        self, node: Node, temperature: float
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Select move based on visit count distribution with temperature"""
        # Handle edge case of single move
        if len(node.children) == 1:
            return next(iter(node.children.keys()))

        # Get visit counts and moves
        visits = np.array(
            [child.visit_count for child in node.children.values()], dtype=np.float64
        )
        moves = list(node.children.keys())

        # Handle zero temperature - select most visited move
        if (
            temperature == 0 or temperature < 1e-8
        ):  # Allow for floating point imprecision
            return moves[np.argmax(visits)]

        # Apply temperature and normalize
        # Use float64 for better numerical stability during power operation
        visits = visits ** (1 / temperature)
        total_visits = np.sum(visits)

        if total_visits == 0:
            # Fallback to uniform distribution if no visits
            probs = np.ones_like(visits) / len(visits)
        else:
            probs = visits / total_visits

        # Ensure probabilities sum to 1 and are valid
        probs = np.clip(probs, 0, 1)
        probs = probs / np.sum(probs)

        # Sample move based on visit count distribution
        return moves[np.random.choice(len(moves), p=probs)]

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
        """Run batch of parallel MCTS simulations with memory cleanup"""
        start_time = time.time()
        MAX_BATCH_TIME = 1.0  # Maximum seconds per batch

        nodes_to_expand = []
        board_states = []
        paths = []
        max_depth_this_batch = 0

        # Selection phase - find nodes to evaluate
        for i in range(batch_size):
            # Check time limit periodically (every 8 simulations)
            if i % 8 == 0 and (time.time() - start_time) > MAX_BATCH_TIME:
                if self.debug:
                    print(f"Batch time limit reached after {i} simulations")
                batch_size = i  # Truncate batch
                break

            node = root
            path = [node]
            depth = 0
            visited = set()  # Track visited positions to detect cycles

            # Keep selecting children until we either:
            # 1. Find an unexpanded node
            # 2. Reach a terminal state
            # 3. Hit max depth
            # 4. Detect a cycle
            while depth < 50:  # Prevent infinite loops
                # Check for cycles
                board_hash = node.board.get_hash()
                if board_hash in visited:
                    if self.debug:
                        print(f"Cycle detected at depth {depth}")
                    break
                visited.add(board_hash)

                # Use cached game over check
                if self._is_game_over(node):
                    break

                if not node.is_expanded:
                    # Found a new leaf - add it for expansion
                    if node not in nodes_to_expand:
                        nodes_to_expand.append(node)
                        board_states.append(node.board.state)
                        paths.append(path)
                    break

                # Node is expanded - select child based on UCB
                move, child = node.select_child(self.config.c_puct)
                if not move or not child:
                    break

                node = child
                path.append(node)
                depth += 1

            # Update statistics
            max_depth_this_batch = max(max_depth_this_batch, depth)
            self.max_path_length = max(self.max_path_length, depth)
            if self.debug:
                self.path_lengths.append(depth)
            paths_buffer[i] = path

        if self.debug:
            elapsed = time.time() - start_time
            if elapsed > 1.0:  # Warn about slow batches
                print(f"WARNING: Slow batch simulation ({elapsed:.3f}s)")

        # Expansion and evaluation phase
        if nodes_to_expand:
            # Batch evaluate new leaf nodes
            board_batch = mx.array(np.stack(board_states), dtype=mx.float32)
            policies, values = self.model(board_batch)

            # Evaluate tensors
            mx.eval(policies, values)

            # Expand new nodes
            for node, policy, value, path in zip(
                nodes_to_expand, policies, values, paths
            ):
                self._expand_node(node, policy, value.item())
                self.backup(path, value.item())

            # Clean up tensors
            del board_batch
            del policies
            del values

        # Always backup values for all paths
        for path in paths_buffer[:batch_size]:
            if not path:
                continue
            leaf_node = path[-1]

            if leaf_node.board.is_game_over():
                # Terminal state - use game result
                value = leaf_node.board.get_game_result()
                self.backup(path, value)
            elif leaf_node.is_expanded:
                # Expanded node - use current value estimate
                self.backup(path, leaf_node.value())

    def should_stop(self, root: Node) -> bool:
        """More conservative early stopping"""
        if not root.children:
            return False

        moves_data = [
            (move, child.visit_count, child.value())
            for move, child in root.children.items()
        ]

        if len(moves_data) <= 1:
            return False

        moves_data.sort(key=lambda x: x[1], reverse=True)
        best_move, best_visits, best_value = moves_data[0]
        second_best = moves_data[1]

        # Need minimum visits before considering stopping
        if root.visit_count < 100 or best_visits < 50:
            return False

        # Only stop if we have clear dominance in both visits and value
        visit_ratio = best_visits / (second_best[1] + 1e-8)
        value_diff = abs(best_value - second_best[2])

        if visit_ratio > 5.0 and value_diff > 0.3:
            if self.debug:
                print(
                    f"Clear dominance - visits ratio: {visit_ratio:.1f}, value diff: {value_diff:.2f}"
                )
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

    def _is_game_over(self, node: Node) -> bool:
        return node.board.is_game_over()

    def _cleanup_tree(self, node: Node):
        """Recursively cleanup the MCTS tree"""
        if not node:
            return

        # Recursively cleanup children
        for child in node.children.values():
            self._cleanup_tree(child)

        # Clear node references
        node.children.clear()
        node.parent = None
        node.board = None  # Remove board reference
