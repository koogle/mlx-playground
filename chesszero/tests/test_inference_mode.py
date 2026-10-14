import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlx.core as mx
import numpy as np

from config.model_config import ModelConfig
from model.self_play import play_single_game
from model.training import Trainer
from model.evaluation import play_evaluation_game_worker
from model.inference import BatchedInference


class TestInferenceMode(unittest.TestCase):
    def setUp(self):
        self.searches = []

    def test_server_predictions_match_eval_without_mutating_source_parameters(self):
        from model.network import ChessNet

        config = ModelConfig(n_residual_blocks=0, n_filters=2, policy_output_dim=4)
        model = ChessNet(config)
        model.eval()
        states = mx.ones((1, 19, 8, 8))
        expected_policy, expected_value = model(states)
        mean, variance = (
            np.array(model.bn_input.running_mean),
            np.array(model.bn_input.running_var),
        )
        with BatchedInference(model, config) as service:
            client = service.client()
            try:
                policy, value = client(states)
            finally:
                client.close()
        np.testing.assert_allclose(policy, expected_policy, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(value, expected_value, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(model.bn_input.running_mean, mean)
        np.testing.assert_array_equal(model.bn_input.running_var, variance)

    def worker_game(self):
        game = Mock()
        game.board.is_game_over.side_effect = [False, False, True]
        game.board.state = np.zeros((19, 8, 8), dtype=np.float32)
        game.board.get_game_result.return_value = 1.0
        game.board.get_current_turn.side_effect = [0, 1]
        return game

    def checking_search(self, model, config, **kwargs):
        search = Mock()
        search.get_move.return_value = ((0, 1), (2, 2))
        search.root_node = SimpleNamespace(
            children={((0, 1), (2, 2)): SimpleNamespace(visit_count=1)}
        )
        self.searches.append(search)
        return search

    def test_self_play_worker_keeps_model_in_eval(self):
        model, connection = Mock(), Mock()
        with (
            patch("model.self_play.ChessGame", return_value=self.worker_game()),
            patch("model.self_play.MCTS", side_effect=self.checking_search),
            patch("model.self_play.tqdm"),
        ):
            config = ModelConfig(temperature=0.7, temp_decay_steps=1, temp_final=0.2)
            play_single_game(model, config, 0, connection, 1)
        self.assertEqual(
            [
                call.kwargs["temperature"]
                for call in self.searches[0].get_move.call_args_list
            ],
            [0.7, 0.2],
        )
        self.assertIsNotNone(connection.send.call_args.args[0])
        model.close.assert_called_once()

    def test_evaluation_worker_sets_both_models_to_eval(self):
        model, best_model, connection = Mock(), Mock(), Mock()
        with (
            patch("model.evaluation.ChessGame", return_value=self.worker_game()),
            patch("model.evaluation.MCTS", side_effect=self.checking_search),
            patch("model.evaluation.tqdm"),
        ):
            play_evaluation_game_worker(
                model, best_model, ModelConfig(), 0, 0, connection, 1
            )
        self.assertIsNotNone(connection.send.call_args.args[0])
        model.close.assert_called_once()
        best_model.close.assert_called_once()

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
