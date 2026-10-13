import multiprocessing as mp
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from chess_engine.bitboard import BitBoard
from config.model_config import ModelConfig
from model.inference import BatchedInference
from model.mcts import MCTS
from test_mcts_sequential import RecordingModel, ToyBoard
from test_shared_inference import AnalyticModel


class TestPredictionReuse(unittest.TestCase):
    def test_deduplication_lru_model_isolation_and_new_snapshot(self):
        first, second = AnalyticModel(), AnalyticModel(offset=10)
        config = ModelConfig(prediction_cache_size=1)
        with BatchedInference([first, second], config) as service:
            clients, connection = service.prepare_job(
                (first, second), mp.get_context("spawn")
            )
            try:
                a, b = clients
                np.testing.assert_array_equal(
                    a(mx.full((3, 19, 8, 8), 2))[1], [[2]] * 3
                )
                a(mx.full((1, 19, 8, 8), 2))
                self.assertEqual(service.positions, 1)
                self.assertEqual(service.cache_hits, 3)
                np.testing.assert_array_equal(b(mx.full((1, 19, 8, 8), 2))[1], [[12]])
                a(mx.full((1, 19, 8, 8), 2))  # Model B evicted A's entry.
                self.assertEqual(service.positions, 3)
                self.assertEqual(len(service.predictions), 1)
            finally:
                connection.close()
        first.offset = 20
        with BatchedInference([first], config) as fresh:
            clients, connection = fresh.prepare_job((first,), mp.get_context("spawn"))
            try:
                np.testing.assert_array_equal(
                    clients[0](mx.full((1, 19, 8, 8), 2))[1], [[22]]
                )
            finally:
                connection.close()

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
