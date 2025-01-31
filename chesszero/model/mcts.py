import math
import time
from random import randint

from tqdm import tqdm
from chess_engine.board import Board, Color
from config.model_config import ModelConfig
from utils.board_utils import encode_board
from typing import List, Tuple
import mlx.core as mx
from chess_engine.bitboard import BitBoard


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

        # Pre-calculate values that are used for all children
        total_visits = max(1, self.visit_count)
        sqrt_total_visits = math.sqrt(total_visits)

        # Use list comprehension and max() instead of iterating
        move, child = max(
            self.children.items(),
            key=lambda x: (
                (-x[1].value() if x[1].visit_count > 0 else 0)
                + c_puct * x[1].prior * sqrt_total_visits / (1 + x[1].visit_count)
            ),
        )
        return move, child


class MCTS:
    def __init__(self, model, config: ModelConfig):
        self.model = model
        self.config = config
        self.debug = True
        self.board_cache = {}
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
            "board_copy": 0.0,  # New: track board copying time
            "move_validation": 0.0,  # New: track move validation time
            "total_moves": 0,
            "total_time": 0.0,
            "board_copies": 0,  # New: count number of board copies
            "move_validations": 0,  # New: count number of move validations
            "total_nodes_expanded": 0,  # New: track total nodes expanded
            "total_inferences": 0,  # New: track total model inferences
        }

    def get_move(self, board: BitBoard):
        """Get the best move for the current position after running MCTS"""
        start_time = time.time()
        self.model.eval()
        try:
            root = Node(board)

            # Initial model inference and expansion
            t0 = time.time()
            policies, values = self.model(board.state[None, ...])
            policy = mx.array(policies[0])
            value = values[0].item()
            self.perf_stats["model_inference"] += time.time() - t0
            self.perf_stats["total_inferences"] += 1

            if self.debug:
                print("\nRoot node activations:")
                print(f"Value: {value:.3f}")
                print(f"Policy stats:")
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

            # Initialize arrays once for all chunks
            batch_size = self.config.n_simulations
            chunk_size = 128
            encoded_boards = []
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
                        board_hash = hash(str(node.board))
                        if board_hash not in self.board_cache:
                            self.board_cache[board_hash] = encode_board(node.board)
                        encoded_boards.append(self.board_cache[board_hash])
                        nodes_to_expand.append(node)
                    search_paths.append((path, node))
                self.perf_stats["selection"] += time.time() - t0

                # Batch evaluate positions
                if encoded_boards:
                    t0 = time.time()
                    encoded_batch = mx.stack(encoded_boards)
                    policies, values = self.model(encoded_batch)
                    self.perf_stats["model_inference"] += time.time() - t0
                    self.perf_stats["total_inferences"] += len(encoded_boards)

                    if self.debug and sim_start == 0:  # Sample first batch
                        print("\nBatch activations sample (first 5):")
                        for i in range(min(5, len(values))):
                            print(f"\nNode {i}:")
                            print(f"Value: {values[i].item():.3f}")
                            print(f"Policy stats:")
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
                    encoded_boards.clear()
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
        current_turn = board.get_current_turn()
        pieces = board.get_all_pieces(current_turn)

        # Get valid moves for each piece
        for pos, piece_type in pieces:
            valid_moves = board.get_valid_moves(pos)
            for to_pos in valid_moves:
                move_idx = self.encode_move(pos, to_pos)
                prior = policy[move_idx]

                child_board = board.copy()
                child_board.make_move(pos, to_pos)

                node.children[(pos, to_pos)] = Node(
                    board=child_board, parent=node, prior=prior
                )

        node.is_expanded = True
        node.value_sum = value
        node.visit_count = 1

    def _print_perf_stats(self):
        """Print performance statistics"""
        total_time = self.perf_stats["total_time"]
        moves = self.perf_stats["total_moves"]
        print("\nPerformance Statistics:")
        print(f"Total moves: {moves}")
        print(f"Average time per move: {total_time/moves:.3f}s")
        print("\nTime breakdown:")
        print(
            f"Model inference: {self.perf_stats['model_inference']/total_time*100:.1f}%"
        )
        print(
            f"Node expansion: {self.perf_stats['node_expansion']/total_time*100:.1f}%"
        )
        print(f"Selection: {self.perf_stats['selection']/total_time*100:.1f}%")
        print(f"Backup: {self.perf_stats['backup']/total_time*100:.1f}%")
        print(f"Board copying: {self.perf_stats['board_copy']/total_time*100:.1f}%")
        print(
            f"Move validation: {self.perf_stats['move_validation']/total_time*100:.1f}%"
        )

        print("\nOperation counts:")
        print(f"Board copies: {self.perf_stats['board_copies']}")
        print(f"Move validations: {self.perf_stats['move_validations']}")
        print(f"Nodes expanded: {self.perf_stats['total_nodes_expanded']}")
        print(f"Model inferences: {self.perf_stats['total_inferences']}")

        if moves > 0:
            print("\nPer move statistics:")
            print(f"Board copies per move: {self.perf_stats['board_copies']/moves:.1f}")
            print(
                f"Move validations per move: {self.perf_stats['move_validations']/moves:.1f}"
            )
            print(
                f"Nodes expanded per move: {self.perf_stats['total_nodes_expanded']/moves:.1f}"
            )
            print(
                f"Model inferences per move: {self.perf_stats['total_inferences']/moves:.1f}"
            )
