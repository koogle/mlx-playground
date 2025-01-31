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
        # Add training flag and different thresholds
        self.training = True
        # Remove pruning during training completely, very small threshold during evaluation
        self.training_prior_threshold = -1.0  # Accept all moves during training
        self.eval_prior_threshold = -1.0  # Accept all moves during evaluation for now
        self.perf_stats = {
            "model_inference": 0.0,
            "node_expansion": 0.0,
            "selection": 0.0,
            "backup": 0.0,
            "total_moves": 0,
            "total_time": 0.0,
            "total_nodes_expanded": 0,
            "total_inferences": 0,
            # Node expansion breakdown
            "get_pieces_time": 0.0,
            "valid_moves_time": 0.0,
            "move_encoding_time": 0.0,
            "node_creation_time": 0.0,
            "board_copy_time": 0.0,
            "make_move_time": 0.0,
        }

    def get_move(self, board: BitBoard):
        """Get the best move for the current position after running MCTS"""
        start_time = time.time()
        self.model.eval()
        try:
            if self.debug:
                print(f"\nBoard state shape: {board.state.shape}")

            root = Node(board)

            # Initial model inference and expansion
            t0 = time.time()
            # Convert numpy array to MLX array with float32 dtype
            board_state = mx.array(board.state, dtype=mx.float32)[None, ...]
            policies, values = self.model(board_state)
            policy = mx.array(policies[0])
            value = values[0].item()
            self.perf_stats["model_inference"] += time.time() - t0
            self.perf_stats["total_inferences"] += 1

            if self.debug:
                print("\nRoot node activations:")
                print(f"Value: {value:.3f}")
                print("Policy stats:")
                print(f"  Min: {mx.min(policy):.3f}")
                print(f"  Max: {mx.max(policy):.3f}")
                print(f"  Mean: {mx.mean(policy):.3f}")
                print(f"  Std: {mx.std(policy):.3f}")

            t0 = time.time()
            self._expand_node(root, policy, value)
            self.perf_stats["node_expansion"] += time.time() - t0
            self.perf_stats["total_nodes_expanded"] += 1

            if not root.children:
                print("\nWarning: Root node has no children after expansion!")
                print(f"Board state:\n{board}")
                print(f"Root value: {value}")
                print(f"Game over: {board.is_game_over()}")
                valid_moves = []
                pieces = board.get_all_pieces(board.get_current_turn())
                for pos, piece_type in pieces:
                    moves = board.get_valid_moves(pos)
                    if moves:
                        valid_moves.extend([(pos, move) for move in moves])
                print(f"Valid moves: {valid_moves}")
                if self.debug:
                    print(f"Policy shape: {policy.shape}")
                    print(f"Policy range: [{mx.min(policy):.4f}, {mx.max(policy):.4f}]")
                return None if not valid_moves else valid_moves[0]

            # Initialize arrays once for all chunks
            batch_size = self.config.n_simulations
            chunk_size = 128
            boards_to_evaluate = []
            nodes_to_expand = []
            search_paths = []

            # Process simulations in chunks
            for sim_start in range(0, batch_size, chunk_size):
                chunk_end = min(sim_start + chunk_size, batch_size)
                current_chunk_size = chunk_end - sim_start

                # Selection phase for current chunk
                t0 = time.time()
                for _ in range(current_chunk_size):
                    node = root
                    path = [node]

                    while node.is_expanded and node.children:
                        move, child = node.select_child(self.config.c_puct)
                        if not move:
                            break
                        node = child
                        path.append(node)

                    if not node.board.is_game_over():
                        # Board state is already in correct format
                        boards_to_evaluate.append(node.board.state)
                        nodes_to_expand.append(node)
                    search_paths.append((path, node))
                self.perf_stats["selection"] += time.time() - t0

                # Batch evaluate positions
                if boards_to_evaluate:
                    t0 = time.time()
                    # Stack board states directly
                    board_batch = mx.array(
                        np.stack(boards_to_evaluate), dtype=mx.float32
                    )
                    policies, values = self.model(board_batch)
                    self.perf_stats["model_inference"] += time.time() - t0
                    self.perf_stats["total_inferences"] += len(boards_to_evaluate)

                    if self.debug and sim_start == 0:  # Sample first batch
                        print("\nBatch activations sample (first 5):")
                        for i in range(min(5, len(values))):
                            print(f"\nNode {i}:")
                            print(f"Value: {values[i].item():.3f}")
                            print("Policy stats:")
                            policy = mx.array(policies[i])
                            print(f"  Min: {mx.min(policy):.3f}")
                            print(f"  Max: {mx.max(policy):.3f}")
                            print(f"  Mean: {mx.mean(policy):.3f}")
                            print(f"  Std: {mx.std(policy):.3f}")

                    t0 = time.time()
                    for node, policy, value in zip(nodes_to_expand, policies, values):
                        policy = mx.array(policy)
                        value = value.item()
                        self._expand_node(node, policy, value)
                    self.perf_stats["node_expansion"] += time.time() - t0
                    self.perf_stats["total_nodes_expanded"] += len(nodes_to_expand)

                    # Clear arrays for next chunk
                    boards_to_evaluate.clear()
                    nodes_to_expand.clear()

                # Backup phase
                t0 = time.time()
                for path, end_node in search_paths:
                    value = (
                        end_node.board.get_game_result()
                        if end_node.board.is_game_over()
                        else end_node.value()
                    )
                    self.backup(path, value)
                self.perf_stats["backup"] += time.time() - t0
                search_paths.clear()

                # Early stopping check
                best_visits = float("-inf")
                second_best = float("-inf")
                best_move = None

                for move, child in root.children.items():
                    visits = child.visit_count
                    if visits > best_visits:
                        second_best = best_visits
                        best_visits = visits
                        best_move = move
                    elif visits > second_best:
                        second_best = visits

                    if best_visits > second_best * 4 and sim_start > batch_size // 2:
                        self.perf_stats["total_moves"] += 1
                        self.perf_stats["total_time"] += time.time() - start_time
                        if self.perf_stats["total_moves"] % 10 == 0:
                            self._print_perf_stats()
                        return best_move

            self.perf_stats["total_moves"] += 1
            self.perf_stats["total_time"] += time.time() - start_time

            if self.perf_stats["total_moves"] % 10 == 0:
                self._print_perf_stats()

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

    def encode_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
        """Encode a move into a policy index"""
        from_idx = from_pos[0] * 8 + from_pos[1]
        to_idx = to_pos[0] * 8 + to_pos[1]
        return from_idx * 64 + to_idx

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

    def _expand_node(self, node: Node, policy, value):
        """Expand node with all valid moves"""
        t0 = time.time()
        board = node.board
        pieces = board.get_all_pieces(board.get_current_turn())
        get_pieces_time = time.time() - t0

        # Convert MLX policy array to numpy for indexing
        policy_np = np.array(policy)

        # Timing variables
        move_encoding_time = 0
        node_creation_time = 0
        copy_time = 0
        make_move_time = 0
        valid_moves_time = 0

        for pos, piece_type in pieces:
            t1 = time.time()
            valid_moves = board.get_valid_moves(
                pos
            )  # Direct call, no caching since it's not helping
            valid_moves_time += time.time() - t1

            for to_pos in valid_moves:
                t1 = time.time()
                move_idx = self.encode_move(pos, to_pos)
                prior = policy_np[move_idx]  # Use numpy array for indexing
                move_encoding_time += time.time() - t1

                t1 = time.time()
                child_board = board.copy()
                copy_time += time.time() - t1

                t1 = time.time()
                child_board.make_move(pos, to_pos)
                make_move_time += time.time() - t1

                t1 = time.time()
                node.children[(pos, to_pos)] = Node(
                    board=child_board, parent=node, prior=prior
                )
                node_creation_time += time.time() - t1

        # Update performance stats
        self.perf_stats["get_pieces_time"] += get_pieces_time
        self.perf_stats["valid_moves_time"] += valid_moves_time
        self.perf_stats["move_encoding_time"] += move_encoding_time
        self.perf_stats["node_creation_time"] += node_creation_time
        self.perf_stats["board_copy_time"] += copy_time
        self.perf_stats["make_move_time"] += make_move_time

        node.is_expanded = True
        node.value_sum = value
        node.visit_count = 1

    def _print_perf_stats(self):
        """Print performance statistics"""

        def safe_percentage(numerator_key: str, denominator: float) -> float:
            """Safely compute percentage, handling missing keys and zero division"""
            if numerator_key not in self.perf_stats or denominator == 0:
                return 0.0
            return self.perf_stats[numerator_key] / denominator * 100

        total_time = self.perf_stats.get("total_time", 0)
        moves = self.perf_stats.get("total_moves", 0)

        print("\nPerformance Statistics:")
        print(f"Total moves: {moves}")
        print(f"Average time per move: {total_time/max(1, moves):.3f}s")

        print("\nTime breakdown:")
        for key in ["model_inference", "node_expansion", "selection", "backup"]:
            pct = safe_percentage(key, total_time)
            print(f"{key.replace('_', ' ').title()}: {pct:.1f}%")

        print("\nOperation counts:")
        for key in ["total_nodes_expanded", "total_inferences"]:
            count = self.perf_stats.get(key, 0)
            print(f"{key.replace('total_', '').replace('_', ' ').title()}: {count}")

        if moves > 0:
            print("\nPer move statistics:")
            for key in ["total_nodes_expanded", "total_inferences"]:
                per_move = self.perf_stats.get(key, 0) / moves
                stat_name = key.replace("total_", "").replace("_", " ").title()
                print(f"{stat_name} per move: {per_move:.1f}")

        print("\nNode expansion breakdown:")
        total_expansion = self.perf_stats.get("node_expansion", 0)
        expansion_keys = [
            "get_pieces_time",
            "valid_moves_time",
            "move_encoding_time",
            "node_creation_time",
            "board_copy_time",
            "make_move_time",
        ]
        for key in expansion_keys:
            pct = safe_percentage(key, total_expansion)
            print(f"{key.replace('_time', '').replace('_', ' ').title()}: {pct:.1f}%")
