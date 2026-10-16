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
from model.inference_mode import search_inference_mode


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

    def test_search_preserves_running_statistics_and_original_mode(self):
        for training, batch_stats in (
            (True, True),
            (False, True),
            (True, False),
            (False, False),
        ):
            with self.subTest(training=training, batch_stats=batch_stats):
                model = ModeModel()
                model.train(training)
                mean = np.array(model.bn.running_mean)
                variance = np.array(model.bn.running_var)
                search = MCTS(
                    model, ModelConfig(n_simulations=4, search_batch_norm=batch_stats)
                )
                search.get_move(BitBoard())
                self.assertTrue(model.modes)
                self.assertFalse(any(model.modes))
                self.assertEqual(model.training, training)
                self.assertTrue(model.bn.track_running_stats)
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
            self.assertTrue(model.bn.track_running_stats)

    def test_batch_statistics_match_manual_normalization_without_updating_buffers(self):
        model = ModeModel()
        inputs = mx.arange(8, dtype=mx.float32).reshape(2, 4, 1) + 10
        for batch_stats in (True, False):
            with self.subTest(batch_stats=batch_stats):
                with search_inference_mode(model, batch_stats):
                    actual = model.bn(inputs)
                    if batch_stats:
                        expected = (inputs - mx.mean(inputs)) / mx.sqrt(
                            mx.var(inputs) + model.bn.eps
                        )
                    else:
                        expected = inputs / mx.sqrt(mx.array(1 + model.bn.eps))
                    np.testing.assert_allclose(actual, expected, atol=1e-6)
                np.testing.assert_array_equal(model.bn.running_mean, [0])
                np.testing.assert_array_equal(model.bn.running_var, [1])
                self.assertTrue(model.bn.track_running_stats)

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

    def test_self_play_worker_uses_cpu_client(self):
        client, connection = Mock(), Mock()
        with (
            patch("model.self_play.ChessGame", return_value=self.worker_game()),
            patch("model.self_play.MCTS") as make_search,
            patch("model.self_play.tqdm"),
        ):
            search = make_search.return_value
            search.get_move.return_value = ((0, 1), (2, 2))
            search.root_node = SimpleNamespace(
                children={((0, 1), (2, 2)): SimpleNamespace(visit_count=1)}
            )
            config = ModelConfig(temperature=0.7, temp_decay_steps=1, temp_final=0.2)
            play_single_game(client, config, 0, connection, 1)
        self.assertEqual(
            [call.kwargs["temperature"] for call in search.get_move.call_args_list],
            [0.7, 0.2],
        )
        self.assertIsNotNone(connection.send.call_args.args[0])
        client.close.assert_called_once()
        client.eval.assert_not_called()

    def test_evaluation_worker_uses_shared_clients_without_noise(self):
        from model.evaluation import play_evaluation_game

        client, connection = Mock(), Mock()
        with (
            patch("model.evaluation.ChessGame", return_value=self.worker_game()),
            patch("model.evaluation.MCTS") as make_search,
        ):
            make_search.return_value.get_move.return_value = ((0, 1), (2, 2))
            play_evaluation_game(client, ModelConfig(), 0, 0, False, 0, connection, 1)
        self.assertEqual(make_search.call_count, 2)
        self.assertTrue(
            all(
                call.kwargs["exploration_noise"] is False
                for call in make_search.call_args_list
            )
        )
        client.for_model.assert_called_once_with(1)
        client.close.assert_called_once()
        client.eval.assert_not_called()
        self.assertIsNotNone(connection.send.call_args.args[0])

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
