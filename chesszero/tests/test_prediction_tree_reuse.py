import unittest
from unittest.mock import patch

import numpy as np

from chess_engine.bitboard import BitBoard
from config.model_config import ModelConfig
from model.mcts import MCTS, Node
from test_mcts_sequential import RecordingModel, ToyBoard


class TestPredictionReuse(unittest.TestCase):
    def test_prediction_identity_uses_nn_input_not_en_passant_state(self):
        model = RecordingModel()
        model.eval()
        search, board = MCTS(model, ModelConfig(search_batch_norm=False)), BitBoard()
        policy, _ = search._evaluate_board(board)
        board.en_passant_target = (2, 0)
        repeated, _ = search._evaluate_board(board)
        self.assertIs(policy, repeated)
        self.assertEqual(len(model.inputs), 1)

    def test_batch_statistics_do_not_reuse_cached_predictions(self):
        model, board = RecordingModel(), BitBoard()
        search = MCTS(model, ModelConfig())
        search._evaluate_board(board)
        search._evaluate_board(board)
        self.assertEqual(len(model.inputs), 2)
        self.assertEqual(len(search._evaluation_cache), 0)


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
        self.assertIs(search.root_node, child)
        self.assertEqual(child.visit_count, visits + 64)

    def test_each_search_shares_one_noise_sample_only_at_the_root(self):
        board = ToyBoard()
        search = MCTS(RecordingModel(), ModelConfig(n_simulations=32))
        selections = []
        select_child = Node.select_child

        def record_selection(node, c_puct, priors=None):
            selections.append((node, priors))
            return select_child(node, c_puct, priors)

        with (
            patch(
                "numpy.random.dirichlet",
                side_effect=[np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            ) as sample,
            patch.object(Node, "select_child", record_selection),
        ):
            for turn in range(2):
                selections.clear()
                move = search.get_move(board)
                root = search.root_node
                root_selections = [
                    priors for node, priors in selections if node is root
                ]
                self.assertEqual(len(root_selections), 32)
                self.assertTrue(any(node is not root for node, _ in selections))
                for node, priors in selections:
                    self.assertIs(priors, root_selections[0] if node is root else None)
                expected = np.full(2, (1 - search.config.dirichlet_epsilon) / 2)
                expected[turn] += search.config.dirichlet_epsilon
                np.testing.assert_allclose(root_selections[0], expected)
                np.testing.assert_allclose(
                    [child.prior for child in root.children.values()], [0.5, 0.5]
                )
                board.make_move(*move)
                search.advance_root(move, board)
            self.assertEqual(sample.call_count, 2)

    def test_history_and_model_changes_drop_tree(self):
        board = BitBoard()
        search = MCTS(RecordingModel(), ModelConfig(n_simulations=0))
        move = search.get_move(board)
        self.assertIsNone(search.root_node.children[move]._board)
        board.make_move(*move)
        search.advance_root(move, board)
        root = search.root_node
        board.position_history[board.get_repetition_hash()] += 1
        search.get_move(board)
        self.assertIsNot(search.root_node, root)
        previous = search.root_node
        search.model_version = 1
        search.get_move(board)
        self.assertIsNot(search.root_node, previous)
