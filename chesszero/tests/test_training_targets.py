import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from config.model_config import ModelConfig
from model.network import ChessNet
from model.self_play import create_batches, generate_games
from model.training import Trainer


def position(white_to_move):
    state = np.zeros((19, 8, 8), dtype=np.float32)
    state[12].fill(white_to_move)
    return mx.array(state)


class TestValueTargets(unittest.TestCase):
    def test_generated_game_has_one_target_per_position(self):
        history = [
            (position(True), mx.array([0.75, 0.25]), None),
            (position(False), mx.array([0.25, 0.75]), None),
        ]
        for white_result in (1.0, -1.0, 0.0):
            with self.subTest(white_result=white_result):
                result = {
                    "game_id": 0,
                    "history": history,
                    "white_result": white_result,
                    "black_result": -white_result,
                    "n_moves": 2,
                    "final_board": "test board",
                }
                with patch(
                    "model.self_play.run_pipe_games", return_value=([result], 0, {})
                ):
                    games = generate_games(
                        None, ModelConfig(n_games_per_iteration=1), max_workers=1
                    )
                self.assertEqual(len(games), 1)
                batches = list(create_batches(games, batch_size=2))
                self.assertEqual(len(batches), 1)
                states, policies, values = batches[0]
                np.testing.assert_array_equal(values, [white_result, -white_result])
                np.testing.assert_array_equal(states, mx.stack([h[0] for h in history]))
                np.testing.assert_array_equal(policies, [[0.75, 0.25], [0.25, 0.75]])

    def test_targets_follow_side_to_move_not_history_index(self):
        # Also covers a game history starting from a Black-to-move position.
        history = [
            (position(False), mx.array([0.5, 0.5]), None),
            (position(True), mx.array([0.5, 0.5]), None),
        ]
        _, _, values = next(create_batches([(history, 1.0)], batch_size=2))
        np.testing.assert_array_equal(values, [-1.0, 1.0])


class TestPolicyLoss(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
            n_residual_blocks=0, n_filters=4, policy_output_dim=4, learning_rate=1e-3
        )
        self.model = ChessNet(self.config)
        self.model.eval()
        self.model.policy_fc.weight = mx.zeros_like(self.model.policy_fc.weight)
        self.model.policy_fc.bias = mx.array([2.0, -1.0, 0.5, -3.0])
        self.states = mx.zeros((2, 19, 8, 8))
        self.targets = mx.array([[0.7, 0.0, 0.3, 0.0], [0.0, 0.4, 0.0, 0.6]])

    def test_inference_still_returns_probabilities(self):
        logits, values = self.model.forward_logits(self.states)
        probabilities, inference_values = self.model(self.states)
        np.testing.assert_allclose(logits, [[2, -1, 0.5, -3]] * 2)
        np.testing.assert_allclose(probabilities, mx.softmax(logits, axis=-1))
        np.testing.assert_allclose(mx.sum(probabilities, axis=-1), [1, 1], atol=1e-6)
        np.testing.assert_array_equal(values, inference_values)

    def test_training_step_uses_logits_for_soft_target_cross_entropy(self):
        logits = np.array([[2.0, -1.0, 0.5, -3.0]] * 2)
        log_probabilities = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
        expected = -np.mean(np.sum(np.array(self.targets) * log_probabilities, axis=-1))
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(self.config, checkpoint_dir=directory)
            trainer.model.load_weights(tree_flatten(self.model.parameters()))
            loss, policy_loss, value_loss, l2_loss = trainer.train_on_batch(
                (self.states, self.targets, mx.zeros(2))
            )
        self.assertAlmostEqual(policy_loss, expected, places=5)
        self.assertTrue(np.isfinite([loss, policy_loss, value_loss, l2_loss]).all())

    def test_model_loss_remains_stable_for_confident_wrong_prediction(self):
        self.model.policy_fc.bias = mx.array([100.0, -100.0, 0.0, 0.0])
        targets = mx.array([[0.0, 1.0, 0.0, 0.0]] * 2)
        _, values = self.model(self.states)
        loss = self.model.loss_fn(self.states, targets, values.squeeze())
        self.assertAlmostEqual(loss.item(), 200.0, places=4)


if __name__ == "__main__":
    unittest.main()
