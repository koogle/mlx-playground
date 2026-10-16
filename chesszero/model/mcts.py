import numpy as np
from typing import List, Tuple, Set, Dict
from chess_engine.bitboard import BitBoard
from cachetools import LRUCache
from config.model_config import ModelConfig
from model.move_encoding import MoveDestination, encode_move
from model.inference_mode import search_inference_mode
from contextlib import nullcontext


class Node:
    """Tree node storing state, prior probabilities, visit counts, and Q-values"""

    def __init__(
        self,
        board: BitBoard = None,
        parent=None,
        prior=0.0,
        exploration_noise=True,
        move=None,
        board_factory=None,
    ):
        self.exploration_noise = (
            parent.exploration_noise if parent else exploration_noise
        )
        self.visit_count = 0
        self.value_sum = 0.0
        self.evaluation = 0.0  # Network value from this node's side-to-move perspective
        self._board = board
        # Passing a factory is a little awkward, but keeps lazy boards tied to the
        # owning search's cache without giving each node the whole MCTS object.
        self._board_factory = parent._board_factory if parent else board_factory
        self.move = move
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.is_expanded = False

    @property
    def board(self):
        """Materialize a child position only when search actually visits it."""
        if self._board is None:
            self._board = self._board_factory(self.parent.board, *self.move)
        return self._board

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count  # already a Python float

    def select_child(self, c_puct, priors=None):
        """Optimized child selection with exploration"""
        if not self.children:
            return None, None

        moves = list(self.children.keys())
        children = list(self.children.values())

        if priors is None:
            priors = np.array([child.prior for child in children])

        visit_counts = np.array([child.visit_count for child in children])
        if not np.any(visit_counts):
            priors = priors / np.sum(priors)
            selected_idx = np.argmax(priors)  # More greedy selection
            return moves[selected_idx], children[selected_idx]

        # Calculate UCB scores with lower temperature
        temperature = 0.5  # Reduced from 1.0 for less exploration
        total_visits = np.sum(visit_counts)
        sqrt_total = np.sqrt(total_visits)

        q_values = np.array([-child.value() for child in children])

        # UCB formula with reduced noise
        exploration = c_puct * priors * sqrt_total / (1 + visit_counts)
        ucb_scores = q_values + temperature * exploration

        if self.exploration_noise:
            ucb_scores += np.random.normal(0, 0.001, size=len(ucb_scores))

        selected_idx = np.argmax(ucb_scores)
        return moves[selected_idx], children[selected_idx]


class GameSearchCache:
    """Bounded reuse for one game's fixed model snapshot; never stores Nodes.

    Advance model_version before changing model weights. Clearing cached entries
    only releases cached data; search trees remain valid for the same weights.
    Self-play workers create a new owner for each game and never train mid-game.
    """

    def __init__(self, model, maxsize=10_000, model_version=0):
        self.model = model
        self.model_version = model_version
        self.all_moves = LRUCache(maxsize=maxsize)
        self.boards = LRUCache(maxsize=maxsize)
        self.positions = LRUCache(maxsize=maxsize)
        self.board_states = LRUCache(maxsize=maxsize)
        self.move_lists = LRUCache(maxsize=maxsize)
        self.evaluations = LRUCache(maxsize=maxsize)
        self.rules = tuple(LRUCache(maxsize=maxsize) for _ in range(3))

    def bind(self, model, model_version=None):
        if model is not self.model:
            raise ValueError("A game cache belongs to one model snapshot")
        if model_version is not None and model_version != self.model_version:
            self.clear()
            self.model_version = model_version

    def clear(self):
        """Release cached entries without invalidating retained search trees."""
        for cache in (
            self.all_moves,
            self.boards,
            self.positions,
            self.board_states,
            self.move_lists,
            self.evaluations,
            *self.rules,
        ):
            cache.clear()


class MCTS:
    """Search a fixed model snapshot; advance model_version when weights change."""

    # Geometry/encoding tables are constant; position caches have explicit owners.
    _move_encoding_table = None  # Will be initialized in __init__

    def __init__(
        self,
        model,
        config: ModelConfig,
        cache=None,
        model_version=None,
        exploration_noise=True,
    ):
        self.cache = cache if cache is not None else GameSearchCache(model)
        self.cache.bind(model, model_version)
        self.model_version = model_version
        self.exploration_noise = exploration_noise
        self._all_moves_cache = self.cache.all_moves
        self._board_cache = self.cache.boards
        self._position_cache = self.cache.positions
        self._board_state_cache = self.cache.board_states
        self._move_list_cache = self.cache.move_lists
        self._evaluation_cache = self.cache.evaluations
        self._rule_caches = self.cache.rules
        self.model = model
        self.config = config
        self.debug = config.debug
        self.training_prior_threshold = -1.0
        self.eval_prior_threshold = -1.0
        self.root_node = None
        self._root_model_version = None
        # Initialize move encoding table if not already done
        if MCTS._move_encoding_table is None:
            MCTS._move_encoding_table = self._init_move_encoding_table()

        if self.debug:
            self.path_lengths = []  # Track all path lengths

    def _init_move_encoding_table(
        self,
    ) -> Dict[Tuple[Tuple[int, int], MoveDestination], int]:
        """Pre-compute move index for every possible (from_pos, to_pos) combination, including pawn promotions"""
        table = {}
        for from_row in range(8):
            for from_col in range(8):
                for to_row in range(8):
                    for to_col in range(8):
                        from_pos = (from_row, from_col)
                        to_pos = (to_row, to_col)
                        table[(from_pos, to_pos)] = encode_move(from_pos, to_pos)

                        # Add promotion moves
                        if (from_row, to_row) in ((6, 7), (1, 0)) and abs(
                            from_col - to_col
                        ) <= 1:
                            for promotion_piece in range(
                                1, 5
                            ):  # Knight, Bishop, Rook, Queen
                                promotion = (to_row, to_col, promotion_piece)
                                table[(from_pos, promotion)] = encode_move(
                                    from_pos, promotion
                                )

        return table

    def encode_move(self, from_pos: Tuple[int, int], to_pos: MoveDestination) -> int:
        """Fast move encoding using lookup table"""
        return MCTS._move_encoding_table[(from_pos, to_pos)]

    def _expand_node(self, node: Node, policy, value):
        """Expand with this evaluation, keeping priors and statistics local to the tree."""
        board_hash = node.board.get_hash()
        node.evaluation = float(value)

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

        board = node.board
        policy_np = np.array(policy)

        moves = self._get_move_list(board)

        if not moves:
            return

        children = {}
        total_prior = 0.0

        for from_pos, to_pos in moves:
            move_idx = MCTS._move_encoding_table[(from_pos, to_pos)]
            prior = max(float(policy_np[move_idx]), 1e-8)
            total_prior += prior
            children[(from_pos, to_pos)] = Node(
                parent=node, prior=prior, move=(from_pos, to_pos)
            )

        # Normalize priors
        if total_prior > 0:
            for child in children.values():
                child.prior /= total_prior

        node.children = children
        node.is_expanded = True

    def _root_search_priors(self, root):
        """Sample once per search, leaving the tree's model priors unchanged."""
        priors = np.array([child.prior for child in root.children.values()])
        if self.exploration_noise and len(priors):
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(priors))
            epsilon = self.config.dirichlet_epsilon
            priors = (1 - epsilon) * priors + epsilon * noise
        return priors

    def _same_position(self, first, second):
        return self._board_key(first) == self._board_key(second) and getattr(
            first, "position_history", None
        ) == getattr(second, "position_history", None)

    def advance_root(self, move, board):
        """Keep the played child's descendants, visits and side-to-move values.

        Materialize before detaching: lazy children still need their parent to
        apply the move. A different path/history must start a fresh tree.
        """
        child = self.root_node.children.get(move) if self.root_node else None
        if child is None or not self._same_position(child.board, board):
            self.root_node = None
            return
        self.root_node.children.clear()  # Release siblings from the old root.
        child.parent = None
        child.move = None
        self.root_node = child

    @staticmethod
    def _board_key(board):
        if isinstance(board, BitBoard):
            return (
                board.get_hash(),
                tuple(sorted(board.king_positions.items())),
                getattr(board, "en_passant_target", None),
            )
        return board.get_hash()

    def _attach_rule_caches(self, board):
        if isinstance(board, BitBoard):
            board._valid_moves_cache, board._pin_cache, board._game_over_cache = (
                self._rule_caches
            )

    def _get_child_board(self, board, source, target, position_key=None):
        # Looking up the transition BEFORE copy/make_move avoids reconstructing
        # successors already built for the same physical parent position.
        # Repetition history is attached separately and never merged between paths.
        key = (
            self._board_key(board) if position_key is None else position_key,
            source,
            target,
        )
        if key not in self._board_cache:
            child = board.copy()
            self._attach_rule_caches(child)
            if not child.make_move(source, target):
                raise ValueError("Cannot cache a board produced by an illegal move")
            if isinstance(child, BitBoard):
                child.freeze()
            self._board_cache[key] = child
            self._position_cache[self._board_key(child)] = child
            return child
        child = self._board_cache[key]
        return (
            child._with_search_history(board) if isinstance(child, BitBoard) else child
        )

    def _get_board_state(self, board):
        """Reuse the network input tensor for a physical position."""
        key = self._board_key(board)
        if key not in self._board_state_cache:
            self._board_state_cache[key] = np.array(board.state, dtype=np.float32)[
                None, ...
            ]
        return self._board_state_cache[key]

    def _get_move_list(self, board):
        """Reuse an immutable expansion skeleton, without Node objects or priors."""
        key = self._board_key(board)
        if key not in self._move_list_cache:
            self._move_list_cache[key] = tuple(
                (source, target)
                for source, targets in self._get_all_valid_moves(board).items()
                for target in targets
            )
        return self._move_list_cache[key]

    def _evaluate_board(self, board):
        """Reuse predictions only when normalization is independent of the batch."""
        inputs = np.asarray(board.state, dtype=np.float32)
        key = (self.cache.model_version, inputs.shape, inputs.tobytes())
        if not self.config.search_batch_norm and key in self._evaluation_cache:
            return self._evaluation_cache[key]
        states = self._get_board_state(board)
        if getattr(self.model, "training", None) is not None:
            # Interactive/local network calls still require MLX arrays. CPU pipe
            # workers keep NumPy inputs and never import MLX.
            import mlx.core as mx

            states = mx.array(states)
        policies, values = self.model(states)
        policy = np.array(policies[0])
        policy.flags.writeable = False
        result = policy, float(values[0].item())
        if not self.config.search_batch_norm:
            self._evaluation_cache[key] = result
        return result

    def _clear_caches(self):
        self.cache.clear()

    def _get_root_board(self, board):
        key = self._board_key(board)
        if key in self._position_cache and isinstance(board, BitBoard):
            return self._position_cache[key]._with_position_history(
                board.position_history.copy()
            )
        root = board.copy()
        self._attach_rule_caches(root)
        if isinstance(root, BitBoard):
            root.freeze()
        self._position_cache[key] = root
        return root

    def get_move(self, board: BitBoard, temperature: float = 0.0):
        # Local callers (e.g. interactive play) may pass a network directly.
        # Pipe clients have no mode: their parent owns normalization state.
        context = (
            search_inference_mode(
                self.model,
                self.config.search_batch_norm,
                self.config.search_batch_norm_per_position,
            )
            if getattr(self.model, "training", None) is not None
            else nullcontext()
        )
        with context:
            return self._search_move(board, temperature)

    def _search_move(self, board: BitBoard, temperature: float = 0.0):
        self.cache.bind(self.model, self.model_version)
        # Reuse is valid only within one fixed model and exact history path.
        if (
            self.root_node is None
            or self._root_model_version != self.cache.model_version
            or not self._same_position(self.root_node.board, board)
        ):
            self.root_node = Node(
                self._get_root_board(board),
                exploration_noise=self.exploration_noise,
                board_factory=self._get_child_board,
            )
            self._root_model_version = self.cache.model_version
        if not self.root_node.is_expanded:
            policy, value = self._evaluate_board(self.root_node.board)
            self._expand_node(self.root_node, policy, value)

        if not self.root_node.children:
            return None

        root_priors = self._root_search_priors(self.root_node)

        # Complete each simulation before selecting the next path.
        for simulations_completed in range(1, self.config.n_simulations + 1):
            self._simulate(self.root_node, root_priors)

            # Check if a move clearly dominates
            if simulations_completed >= 70 and self.should_stop(self.root_node):
                if self.debug:
                    print(
                        f"Stopping simulations early at simulation {simulations_completed} of {self.config.n_simulations}"
                    )
                break

        # Select move based on temperature
        move = self._select_move_with_temperature(self.root_node, temperature)
        return move

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
        self._attach_rule_caches(board)
        board_hash = board.get_hash()
        if board_hash in self._all_moves_cache:
            return self._all_moves_cache[board_hash]

        moves = {}
        pieces = board.get_all_pieces(board.get_current_turn())
        for pos, piece in pieces:
            valid_moves = board.get_valid_moves(pos)
            if valid_moves:
                moves[pos] = valid_moves

        self._all_moves_cache[board_hash] = moves
        return moves

    def _simulate(self, root: Node, root_priors=None):
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
                policy, value = self._evaluate_board(node.board)
                self._expand_node(node, policy, value)
                value = node.evaluation
                break

            if len(path) - 1 >= 50 or not node.children:
                value = node.evaluation
                break

            _, node = node.select_child(
                self.config.c_puct, root_priors if node is root else None
            )
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
