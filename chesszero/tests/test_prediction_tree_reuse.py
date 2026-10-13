import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from chess_engine.bitboard import BitBoard
from config.model_config import ModelConfig
from model.inference import BatchedInference
from model.mcts import MCTS
from test_mcts_sequential import RecordingModel, ToyBoard
from test_shared_inference import small_model


class TestPredictionReuse(unittest.TestCase):
    def test_deduplication_lru_model_isolation_and_new_snapshot(self):
        first, config = small_model()
        second, _ = small_model(2)
        config.prediction_cache_size = 1
        state = mx.ones((1, 19, 8, 8))
        with BatchedInference(first, config, reference_model=second) as service:
            a, b = service.client(), service.reference_client()
            expected = np.array(a(state)[0])
            np.testing.assert_array_equal(a(state)[0], expected)
            self.assertEqual(service.rpc.statistics()["positions"], 1)
            self.assertEqual(service.rpc.statistics()["cache_hits"], 1)
            self.assertFalse(np.array_equal(b(state)[0], expected))
            a(state)  # Reference weights evicted the current entry.
            self.assertEqual(service.rpc.statistics()["positions"], 3)
        first.policy_fc.bias = mx.array([3.0, 0.0, 0.0, 0.0])
        with BatchedInference(first, config) as fresh:
            self.assertFalse(np.array_equal(fresh.client()(state)[0], expected))

    def test_prediction_identity_uses_nn_input_not_en_passant_state(self):
        model = RecordingModel()
        model.eval()
        search, board = MCTS(model, ModelConfig()), BitBoard()
        policy, _ = search._evaluate_board(board)
        board.en_passant_target = (2, 0)
        repeated, _ = search._evaluate_board(board)
        self.assertIs(policy, repeated)
        self.assertEqual(len(model.inputs), 1)


class TestTreeReuse(unittest.TestCase):
    def test_promoted_child_keeps_statistics_and_receives_new_simulations(self):
        board, model = ToyBoard(), RecordingModel()
        search = MCTS(model, ModelConfig(n_simulations=32), exploration_noise=False)
        move = search.get_move(board)
        child = search.root_node.children[move]
        visits, value, descendants = child.visit_count, child.value_sum, child.children
        board.make_move(*move)
        search.advance_root(move, board)
        self.assertIs(search.root_node, child)
        self.assertIsNone(child.parent)
        self.assertIs(child.children, descendants)
        self.assertEqual((child.visit_count, child.value_sum), (visits, value))
        search.get_move(board)
        self.assertEqual(child.visit_count, visits + 32)
        search.cache.clear()
        search.get_move(board)
        self.assertIsNot(search.root_node, child)
        self.assertEqual(search.root_node.visit_count, 32)

    def test_fresh_noise_does_not_accumulate_and_history_mismatch_drops_tree(self):
        board = BitBoard()
        search = MCTS(RecordingModel(), ModelConfig(n_simulations=0))
        move = search.get_move(board)
        self.assertIsNone(search.root_node.children[move]._board)
        board.make_move(*move)
        search.advance_root(move, board)
        root = search.root_node
        search.get_move(board)
        with patch(
            "numpy.random.dirichlet",
            return_value=np.ones(len(root.children)) / len(root.children),
        ):
            search._apply_root_noise(root)
            priors = [child.prior for child in root.children.values()]
            search._apply_root_noise(root)
            self.assertEqual(priors, [child.prior for child in root.children.values()])
        board.position_history[board.get_repetition_hash()] += 1
        search.get_move(board)
        self.assertIsNot(search.root_node, root)
        previous = search.root_node
        search.model_version = 1
        search.get_move(board)
        self.assertIsNot(search.root_node, previous)
