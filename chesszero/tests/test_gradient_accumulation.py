import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from config.model_config import ModelConfig
from model.training import Trainer


class TinyNet(nn.Module):
    """No BatchNorm: split batches should match a single full batch."""

    def __init__(self):
        super().__init__()
        self.policy, self.value = nn.Linear(2, 3), nn.Linear(2, 1)

    def forward_logits(self, states):
        return self.policy(states), mx.tanh(self.value(states))


class TestGradientAccumulation(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        config = ModelConfig(n_epochs=1, eval_interval_epochs=100)
        with patch("model.training.ChessNet", side_effect=lambda config: TinyNet()):
            self.full = Trainer(config, checkpoint_dir=directory.name)
            self.split = Trainer(config, checkpoint_dir=directory.name)
        self.split.model.load_weights(tree_flatten(self.full.model.parameters()))
        self.batch = (
            mx.array([[1.0, 0.0], [0.0, 2.0], [-3.0, 1.0]]),
            mx.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.8, 0.0]]),
            mx.array([1.0, -1.0, 0.5]),
        )
        self.batches = [
            tuple(a[:2] for a in self.batch),
            tuple(a[2:] for a in self.batch),
        ]

    def test_uneven_batches_match_full_batch_loss_weights_and_adam(self):
        for step in (1, 2):
            full_losses = self.full.train_on_batch(self.batch)
            split_losses = self.split.train_on_batches(self.batches)
            np.testing.assert_allclose(full_losses, split_losses, rtol=1e-5)
            for first, second in (
                (self.full.model.parameters(), self.split.model.parameters()),
                (self.full.optimizer.state, self.split.optimizer.state),
            ):
                first, second = dict(tree_flatten(first)), dict(tree_flatten(second))
                self.assertEqual(first.keys(), second.keys())
                for key in first:
                    np.testing.assert_allclose(
                        first[key], second[key], rtol=1e-5, atol=1e-7
                    )
            self.assertEqual(int(self.split.optimizer.state["step"]), step)

    def test_training_steps_scale_with_fresh_positions(self):
        history = [
            (mx.zeros((19, 8, 8)), mx.array([1.0, 0.0, 0.0]), None),
            (mx.zeros((19, 8, 8)), mx.array([0.0, 1.0, 0.0]), None),
            (mx.zeros((19, 8, 8)), mx.array([0.2, 0.8, 0.0]), None),
        ]
        config = self.split.config
        config.policy_output_dim = 3
        config.batch_size = 2
        config.replay_sample_ratio = 2.0  # ceil(2 * 3 fresh / 2) = 3 steps
        config.batchnorm_recalibration_batches = 0
        self.split.replay.policy_output_dim = 3
        with (
            patch("model.training.generate_games", return_value=[(history, 1.0)]),
            patch.object(TinyNet, "forward_logits", self.flat_forward),
        ):
            self.split.train(n_epochs=1)
        self.assertEqual(len(self.split.replay), 3)
        self.assertEqual(int(self.split.optimizer.state["step"]), 3)
        # A second generation of the same size takes the same number of steps
        # even though the buffer has doubled.
        with (
            patch("model.training.generate_games", return_value=[(history, 1.0)]),
            patch.object(TinyNet, "forward_logits", self.flat_forward),
        ):
            self.split.train(n_epochs=1)
        self.assertEqual(len(self.split.replay), 6)
        self.assertEqual(int(self.split.optimizer.state["step"]), 6)

    def test_fresh_position_loss_is_measured_in_eval_mode_without_updating(self):
        history = [
            (mx.zeros((19, 8, 8)), mx.array([1.0, 0.0, 0.0]), None),
            (mx.ones((19, 8, 8)), mx.array([0.0, 1.0, 0.0]), None),
        ]
        self.split.config.policy_output_dim = 3
        before = dict(tree_flatten(self.split.model.parameters()))
        with patch.object(TinyNet, "forward_logits", self.flat_forward):
            losses = self.split.evaluate_fresh_positions([(history, -1.0)])
        self.assertEqual(len(losses), 4)
        self.assertGreater(losses[1], 0.0)
        for key, value in tree_flatten(self.split.model.parameters()):
            np.testing.assert_array_equal(value, before[key])

    def test_empty_generation_skips_training_without_failing(self):
        with patch("model.training.generate_games", return_value=[]):
            self.split.train(n_epochs=1)
        self.assertEqual(int(self.split.optimizer.state["step"]), 0)

    def test_target_signal_summary(self):
        history = [
            (mx.zeros((19, 8, 8)), mx.array([0.5, 0.5, 0.0]), None),  # uniform
            (mx.ones((19, 8, 8)), mx.array([1.0, 0.0, 0.0]), None),  # one-hot
        ]
        self.split.config.policy_output_dim = 3
        with patch.object(TinyNet, "forward_logits", self.flat_forward):
            signal = self.split.describe_targets([(history, 1.0), (history, -1.0)])
        # Two-move uniform rows have relative entropy 1; one-hot rows are skipped.
        self.assertAlmostEqual(signal["policy_relative_entropy"], 1.0)
        # Targets are +-1 with variance 1; an untrained head cannot beat constant.
        self.assertLessEqual(signal["value_explained_variance"], 0.05)

    @staticmethod
    def flat_forward(model, states, original=TinyNet.forward_logits):
        # TinyNet takes two features; the loop feeds full board tensors.
        return original(model, states.reshape(len(states), -1)[:, :2])
