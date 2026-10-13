import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from chess_engine.bitboard import BitBoard
from config.model_config import ModelConfig
from model.mcts import MCTS, Node


class ToyBoard:
    """Small deterministic alternating-turn game with unique position identities."""

    def __init__(self, width=2, terminal_depth=6, terminal_results=None):
        self.width = width
        self.terminal_depth = terminal_depth
        self.terminal_results = terminal_results or {}
        self.position = 1
        self.depth = 0

    @property
    def state(self):
        state = np.zeros((19, 8, 8), dtype=np.float32)
        state[0, 0, :2] = self.position, self.depth
        state[12].fill(1 - self.get_current_turn())
        return state

    def copy(self):
        board = ToyBoard(self.width, self.terminal_depth, self.terminal_results)
        board.position, board.depth = self.position, self.depth
        return board

    def get_hash(self):
        return self.position

    def get_current_turn(self):
        return self.depth % 2

    def get_all_pieces(self, color):
        return [((0, 0), 0)]

    def get_valid_moves(self, source):
        return {(0, index + 1) for index in range(self.width)}

    def make_move(self, source, target):
        self.position = self.position * 3 + target[1]
        self.depth += 1
        return True

    def is_game_over(self):
        return self.depth >= self.terminal_depth

    def get_game_result(self):
        return (
            self.terminal_results.get(self.position, 0.0)
            if self.is_game_over()
            else None
        )


class RecordingModel:
    def __init__(self, fail_on_call=None):
        self.inputs = []
        self.fail_on_call = fail_on_call
        self.training = True

    def __call__(self, states):
        self.inputs.append(np.array(states))
        if len(self.inputs) == self.fail_on_call:
            raise RuntimeError("inference failure")
        return mx.ones((len(states), 4672)) / 4672, mx.full((len(states), 1), 0.25)

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.train(False)


class TestMCTSBatching(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        MCTS._all_moves_cache.clear()

    def search(self, board, simulations=17, batch_size=8, model=None):
        model = model or RecordingModel()
        config = ModelConfig(n_simulations=simulations, debug=True)
        search = MCTS(model, config)
        search.max_batch_size = batch_size
        with patch.object(search, "should_stop", return_value=False):
            move = search.get_move(board)
        return search, model, move

    def test_search_revisits_and_deepens_with_exact_visit_budget(self):
        search, model, _ = self.search(ToyBoard())
        root = search.root_node
        self.assertEqual(root.visit_count, 17)
        self.assertEqual(sum(child.visit_count for child in root.children.values()), 17)
        self.assertGreater(max(search.path_lengths), 1)
        evaluated = [
            int(position) for batch in model.inputs for position in batch[:, 0, 0, 0]
        ]
        self.assertEqual(len(evaluated), len(set(evaluated)))
        self.assertTrue(all(len(batch) <= 8 for batch in model.inputs))
        self.assertTrue(any(len(batch) > 1 for batch in model.inputs))
        stack = [root]
        while stack:
            node = stack.pop()
            self.assertEqual(node.pending_visits, 0)
            stack.extend(node.children.values())

    def test_batch_size_one_preserves_sequential_search(self):
        search, model, _ = self.search(ToyBoard(), simulations=12, batch_size=1)
        self.assertEqual(search.root_node.visit_count, 12)
        self.assertGreater(max(search.path_lengths), 1)
        self.assertTrue(all(len(batch) == 1 for batch in model.inputs))

    def test_forced_line_flushes_pending_leaf_and_continues(self):
        search, model, _ = self.search(
            ToyBoard(width=1, terminal_depth=20), simulations=10
        )
        self.assertEqual(search.root_node.visit_count, 10)
        self.assertEqual(max(search.path_lengths), 10)
        self.assertEqual(
            len(model.inputs), 11
        )  # Root plus ten progressively deeper leaves.

    def test_terminal_revisits_use_exact_result_without_neural_evaluation(self):
        board = ToyBoard(terminal_depth=1, terminal_results={4: -1.0, 5: 1.0})
        search, model, move = self.search(board, simulations=32)
        self.assertEqual(move, ((0, 0), (0, 1)))
        self.assertEqual(len(model.inputs), 1)  # Only the nonterminal root.
        root = search.root_node
        self.assertEqual(root.visit_count, 32)
        winning = root.children[((0, 0), (0, 1))]
        losing = root.children[((0, 0), (0, 2))]
        self.assertEqual(winning.value(), -1.0)
        if losing.visit_count:
            self.assertEqual(losing.value(), 1.0)
        self.assertEqual(root.value_sum, winning.visit_count - losing.visit_count)

    def test_expansion_does_not_count_as_a_completed_simulation(self):
        search = MCTS(RecordingModel(), ModelConfig())
        root = Node(ToyBoard())
        search._expand_node(root, mx.ones(4672) / 4672, 0.75)
        self.assertEqual(root.visit_count, 0)
        self.assertEqual(root.value_sum, 0.0)

    def test_pending_visits_are_released_on_inference_error(self):
        model = RecordingModel(fail_on_call=2)
        search = MCTS(model, ModelConfig(n_simulations=16))
        with self.assertRaisesRegex(RuntimeError, "inference failure"):
            search.get_move(ToyBoard())
        root = search.root_node
        self.assertEqual(root.pending_visits, 0)
        self.assertEqual(root.visit_count, 0)
        for child in root.children.values():
            self.assertEqual(child.pending_visits, 0)
            self.assertEqual(child.visit_count, 0)

    def test_early_stopping_happens_before_full_budget(self):
        model = RecordingModel()
        search = MCTS(model, ModelConfig(n_simulations=100))
        with patch.object(search, "should_stop", return_value=True):
            search.get_move(ToyBoard())
        self.assertGreater(search.root_node.visit_count, 0)
        self.assertLess(search.root_node.visit_count, 100)

    def test_real_chess_opening_deepens_and_preserves_input_board(self):
        board = BitBoard()
        original = board.state.copy()
        search, _, move = self.search(board, simulations=32, batch_size=4)
        self.assertEqual(search.root_node.visit_count, 32)
        self.assertGreater(max(search.path_lengths), 1)
        self.assertIn(move[1], board.get_valid_moves(move[0]))
        np.testing.assert_array_equal(board.state, original)

    def test_real_chess_search_prefers_mate_for_black(self):
        board = BitBoard()
        for source, target in (((1, 5), (2, 5)), ((6, 4), (4, 4)), ((1, 6), (3, 6))):
            self.assertTrue(board.make_move(source, target))
        search, _, move = self.search(board, simulations=96)
        self.assertEqual(move, ((7, 3), (3, 7)))  # ...Qh4# in Fool's Mate.
        self.assertEqual(search.root_node.children[move].value(), -1.0)


if __name__ == "__main__":
    unittest.main()
