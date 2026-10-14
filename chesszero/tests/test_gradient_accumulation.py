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

    def test_training_loop_updates_once_for_the_generation(self):
        def streamed_batches():
            yield self.batches[0]
            # A list() in the outer loop consumes this while model is in eval.
            self.assertTrue(self.split.model.training)
            yield self.batches[1]

        with (
            patch("model.training.generate_games", return_value=[]),
            patch.object(self.split, "_save_checkpoint") as save,
            patch("model.training.create_batches", return_value=streamed_batches()),
        ):
            self.split.train(n_epochs=1)
        self.assertEqual(int(self.split.optimizer.state["step"]), 1)
        save.assert_called_once_with(1)
