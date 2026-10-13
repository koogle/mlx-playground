import numpy as np
from typing import List, Tuple, Set, Dict
import mlx.core as mx
from chess_engine.bitboard import BitBoard
from config.model_config import ModelConfig


class Node:
    """Tree node storing state, prior probabilities, visit counts, and Q-values"""

    def __init__(self, board: BitBoard, parent=None, prior=0.0):
        self.visit_count = 0
        self.value_sum = 0.0
        self.evaluation = 0.0  # Network value from this node's side-to-move perspective
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

        visit_counts = np.array([child.visit_count for child in children])
        if not np.any(visit_counts):
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
        total_visits = np.sum(visit_counts)
        sqrt_total = np.sqrt(total_visits)

        q_values = np.array([-child.value() for child in children])

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
    # Class-level caches
    _position_cache = {}
    _all_moves_cache = {}
    _value_cache = {}  # Cache for evaluated positions
    _policy_cache = {}  # Cache for policy evaluations
    _tree_cache = {}  # Cache for subtrees
    _board_state_cache = {}  # Cache for board states
    _move_list_cache = {}  # Cache for move lists
    _transposition_table = {}  # Store equivalent positions
    _move_encoding_table = None  # Will be initialized in __init__

    def __init__(self, model, config: ModelConfig):
        self.model = model
        self.config = config
        self.debug = config.debug
        self.cache_max_size = 100000  # Adjust this based on memory constraints
        self.training = True
        self.training_prior_threshold = -1.0
        self.eval_prior_threshold = -1.0
        self.root_node = None  # Initialize root_node
        # Initialize move encoding table if not already done
        if MCTS._move_encoding_table is None:
            MCTS._move_encoding_table = self._init_move_encoding_table()

        if self.debug:
            self.path_lengths = []  # Track all path lengths

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
        return MCTS._move_encoding_table[(from_pos, to_pos)]

    def _expand_node(self, node: Node, policy, value):
        """Expand node with better caching"""
        board_hash = node.board.get_hash()
        node.evaluation = float(value)

        # Check transposition table first
        if board_hash in MCTS._transposition_table:
            cached_node = MCTS._transposition_table[board_hash]
            if cached_node.visit_count > node.visit_count:
                node.children = cached_node.children
                node.is_expanded = True
                node.value_sum = cached_node.value_sum
                node.visit_count = cached_node.visit_count
                return

        # Check for cycles
        current = node
        path_positions = {board_hash}
        while current.parent:
            parent_hash = current.parent.board.get_hash()
            if parent_hash in path_positions:
                node.is_expanded = True
                node.evaluation = 0.0
                return
            path_positions.add(parent_hash)
            current = current.parent

        # Use cached subtree if available
        if board_hash in MCTS._tree_cache:
            node.children = MCTS._tree_cache[board_hash]
            node.is_expanded = True
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
                move_idx = MCTS._move_encoding_table[(from_pos, to_pos)]
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

        if len(node.children) > 0:
            MCTS._tree_cache[board_hash] = node.children

        # Cache the expanded node
        # if node.visit_count > 10:  # Only cache well-explored nodes
        #    MCTS._transposition_table[board_hash] = node

    def get_move(self, board: BitBoard, temperature: float = 0.0):
        # Don't switch to eval mode during training
        if not self.training:
            self.model.eval()

        try:
            # Clean caches periodically
            if len(MCTS._all_moves_cache) > self.cache_max_size:
                self._prune_caches()

            key = board.get_hash()
            if key in MCTS._transposition_table:
                cached_node = MCTS._transposition_table[key]
                if cached_node.visit_count > self.config.n_simulations // 2:
                    return self._select_move_with_temperature(cached_node, temperature)

            # Create root and do initial evaluation
            self.root_node = Node(board)
            board_state = mx.array(board.state, dtype=mx.float32)[None, ...]
            policies, values = self.model(board_state)
            self._expand_node(self.root_node, policies[0], values[0].item())

            if not self.root_node.children:
                return None

            # Complete each simulation before selecting the next path.
            for simulations_completed in range(1, self.config.n_simulations + 1):
                self._simulate(self.root_node)

                # Check if a move clearly dominates
                if self.should_stop(self.root_node):
                    if self.debug:
                        print(
                            f"Stopping simulations early at simulation {simulations_completed} of {self.config.n_simulations}"
                        )
                    break

            # Select move based on temperature
            move = self._select_move_with_temperature(self.root_node, temperature)
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

    def backup(self, search_path: List[Node], value: float, value_player: int):
        """Accumulate value in each node's side-to-move perspective.

        value_player identifies whose outcome value estimates (0=White, 1=Black).
        """
        for node in search_path:
            node_value = (
                value if node.board.get_current_turn() == value_player else -value
            )
            node.visit_count += 1
            node.value_sum += node_value

    def _get_all_valid_moves(
        self, board: BitBoard
    ) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
        """Get all valid moves for current player with caching"""
        board_hash = board.get_hash()
        if board_hash in MCTS._all_moves_cache:
            return MCTS._all_moves_cache[board_hash]

        moves = {}
        pieces = board.get_all_pieces(board.get_current_turn())
        for pos, piece in pieces:
            valid_moves = board.get_valid_moves(pos)
            if valid_moves:
                moves[pos] = valid_moves

        MCTS._all_moves_cache[board_hash] = moves
        return moves

    def _simulate(self, root: Node):
        """Select, evaluate, expand and back up one complete simulation."""
        node = root
        path = [node]
        visited = set()

        while True:
            board_hash = node.board.get_hash()
            if board_hash in visited:
                value = 0.0
                break
            visited.add(board_hash)

            if node.board.is_game_over():
                value = node.board.get_game_result()
                if value is None:
                    raise ValueError("Terminal board has no game result")
                break

            if not node.is_expanded:
                state = mx.array(node.board.state, dtype=mx.float32)[None, ...]
                policies, values = self.model(state)
                self._expand_node(node, policies[0], values[0].item())
                value = node.evaluation
                break

            if len(path) - 1 >= 50 or not node.children:
                value = node.evaluation
                break

            _, node = node.select_child(self.config.c_puct)
            path.append(node)

        if self.debug:
            self.path_lengths.append(len(path) - 1)
        self.backup(path, value, value_player=node.board.get_current_turn())

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
        _, best_visits, best_value = moves_data[0]
        second_best = moves_data[1]

        # Need minimum visits before considering stopping
        if root.visit_count < 70 or best_visits < 35:
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

    def _prune_caches(self):
        """More aggressive cache pruning"""
        target_size = int(self.cache_max_size * 0.75)

        # Sort caches by access time and keep most recent
        for cache in [
            MCTS._board_state_cache,
            MCTS._move_list_cache,
            MCTS._value_cache,
            MCTS._policy_cache,
        ]:
            if len(cache) > target_size:
                sorted_items = sorted(
                    cache.items(), key=lambda x: x[1].get("last_access", 0)
                )
                cache.clear()
                cache.update(dict(sorted_items[-target_size:]))

        # Clear older transpositions
        if len(MCTS._transposition_table) > target_size:
            sorted_nodes = sorted(
                MCTS._transposition_table.items(), key=lambda x: x[1].visit_count
            )
            MCTS._transposition_table = dict(sorted_nodes[-target_size:])

        if self.debug:
            print(f"Pruned caches to {target_size} entries")

    def _get_board_state(self, board: BitBoard) -> np.ndarray:
        """Get cached board state array"""
        board_hash = board.get_hash()
        if board_hash not in MCTS._board_state_cache:
            MCTS._board_state_cache[board_hash] = board.state.copy()
        return MCTS._board_state_cache[board_hash]

    def _get_move_list(
        self, board: BitBoard
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get cached list of all possible moves"""
        board_hash = board.get_hash()
        if board_hash not in MCTS._move_list_cache:
            moves = []
            for from_pos, valid_moves in self._get_all_valid_moves(board).items():
                moves.extend((from_pos, to_pos) for to_pos in valid_moves)
            MCTS._move_list_cache[board_hash] = moves
        return MCTS._move_list_cache[board_hash]
