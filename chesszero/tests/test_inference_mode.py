import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from chess_engine.bitboard import BitBoard
from config.model_config import ModelConfig
from model.mcts import MCTS
from model.self_play import play_single_game
from model.training import Trainer


class ModeModel(nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.bn = nn.BatchNorm(1)
        self.modes = []
        self.fail = fail

    def __call__(self, states):
        self.modes.append(self.training)
        if self.fail:
            raise RuntimeError("prediction failed")
        normalized = self.bn(states.reshape(len(states), -1, 1))
        value = mx.tanh(mx.mean(normalized, axis=1))
        return mx.ones((len(states), 4672)) / 4672, value


class TestInferenceMode(unittest.TestCase):
    def setUp(self):
        self.searches = []

    def test_search_uses_eval_and_restores_original_mode(self):
        for training in (True, False):
            with self.subTest(training=training):
                model = ModeModel()
                model.train(training)
                mean = np.array(model.bn.running_mean)
                variance = np.array(model.bn.running_var)
                search = MCTS(model, ModelConfig(n_simulations=4))
                search.get_move(BitBoard())
                self.assertTrue(model.modes)
                self.assertFalse(any(model.modes))
                self.assertEqual(model.training, training)
                np.testing.assert_array_equal(model.bn.running_mean, mean)
                np.testing.assert_array_equal(model.bn.running_var, variance)

    def test_search_restores_original_mode_on_failure(self):
        for training in (True, False):
            model = ModeModel(fail=True)
            model.train(training)
            with self.assertRaisesRegex(RuntimeError, "prediction failed"):
                MCTS(model, ModelConfig()).get_move(BitBoard())
            self.assertEqual(model.modes, [False])
            self.assertEqual(model.training, training)

    def worker_game(self):
        game = Mock()
        game.board.is_game_over.side_effect = [False, False, True]
        game.board.state = np.zeros((19, 8, 8), dtype=np.float32)
        game.board.get_game_result.return_value = 1.0
        game.board.get_current_turn.side_effect = [0, 1]
        return game

    def checking_search(self, model, config, **kwargs):
        self.assertFalse(model.training)
        search = Mock()
        search.get_move.return_value = ((0, 1), (2, 2))
        search.root_node = SimpleNamespace(
            children={((0, 1), (2, 2)): SimpleNamespace(visit_count=1)}
        )
        self.searches.append(search)
        return search

    def test_self_play_worker_keeps_model_in_eval(self):
        model, connection = ModeModel(), Mock()
        with (
            patch("model.self_play.ChessGame", return_value=self.worker_game()),
            patch("model.self_play.MCTS", side_effect=self.checking_search),
            patch("model.self_play.tqdm"),
        ):
            config = ModelConfig(temperature=0.7, temp_decay_steps=1, temp_final=0.2)
            play_single_game(model, config, 0, connection, 1)
        self.assertEqual(
            [
                search.get_move.call_args.kwargs["temperature"]
                for search in self.searches
            ],
            [0.7, 0.2],
        )
        self.assertIsNotNone(connection.send.call_args.args[0])
        self.assertFalse(model.training)

    def test_evaluation_worker_sets_both_models_to_eval(self):
        model, best_model, connection = ModeModel(), ModeModel(), Mock()
        with (
            patch("model.training.ChessGame", return_value=self.worker_game()),
            patch("model.training.MCTS", side_effect=self.checking_search),
            patch("model.training.tqdm"),
        ):
            Trainer.play_evaluation_game_worker(
                model, best_model, ModelConfig(), 0, 0, connection, 1
            )
        self.assertIsNotNone(connection.send.call_args.args[0])
        self.assertFalse(model.training)
        self.assertFalse(best_model.training)

    def test_fresh_network_has_unit_batchnorm_running_variance(self):
        from model.network import ChessNet
        from mlx.utils import tree_flatten

        model = ChessNet(
            ModelConfig(n_residual_blocks=1, n_filters=4, policy_output_dim=4)
        )
        variances = [
            value
            for name, value in tree_flatten(model.parameters())
            if name.endswith("running_var")
        ]
        self.assertEqual(len(variances), 5)
        for variance in variances:
            np.testing.assert_array_equal(variance, np.ones(variance.shape))

    def test_loss_computation_switches_back_to_training_mode(self):
        config = ModelConfig(n_residual_blocks=0, n_filters=2, policy_output_dim=4)
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(config, checkpoint_dir=directory)
            trainer.model.eval()
            trainer.train_on_batch(
                (mx.ones((2, 19, 8, 8)), mx.ones((2, 4)) / 4, mx.zeros(2))
            )
            self.assertTrue(trainer.model.training)


if __name__ == "__main__":
    unittest.main()
