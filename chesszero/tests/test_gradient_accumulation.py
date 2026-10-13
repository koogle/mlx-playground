import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from config.model_config import ModelConfig
from model.self_play import create_batches
from model.training import Trainer


class TinyNet(nn.Module):
    """No BatchNorm, so microbatching must match a full-batch gradient exactly."""

    def __init__(self):
        super().__init__()
        self.policy = nn.Linear(2, 3)
        self.value = nn.Linear(2, 1)

    def forward_logits(self, states):
        return self.policy(states), mx.tanh(self.value(states))


class TestGradientAccumulation(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.batch = (
            mx.array([[1.0, 0.0], [0.0, 2.0], [-3.0, 1.0], [2.0, -2.0], [0.5, 0.3]]),
            mx.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.2, 0.8, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            mx.array([1.0, -1.0, 0.5, 0.0, -0.5]),
        )

    def trainer(self, model=None):
        model = model or TinyNet()
        config = ModelConfig(n_epochs=1, eval_interval_epochs=100, learning_rate=1e-3)
        with patch("model.training.ChessNet", return_value=model):
            return Trainer(config, checkpoint_dir=self.directory.name)

    def microbatches(self):
        return [
            tuple(array[:2] for array in self.batch),
            tuple(array[2:] for array in self.batch),
        ]

    def assert_trees_close(self, first, second):
        first, second = dict(tree_flatten(first)), dict(tree_flatten(second))
        self.assertEqual(first.keys(), second.keys())
        for key in first:
            np.testing.assert_allclose(
                first[key], second[key], rtol=1e-5, atol=1e-7, err_msg=key
            )

    def test_uneven_microbatches_match_full_batch_update_and_adam_moments(self):
        full = self.trainer()
        model = TinyNet()
        model.load_weights(tree_flatten(full.model.parameters()))
        split = self.trainer(model)
        for generation in range(2):
            with self.subTest(generation=generation):
                full_losses = full.train_on_batch(self.batch)
                split_losses = split.train_on_batches(self.microbatches())
                self.assert_trees_close(
                    full.model.parameters(), split.model.parameters()
                )
                self.assert_trees_close(full.optimizer.state, split.optimizer.state)
                np.testing.assert_allclose(full_losses, split_losses, rtol=1e-5)
                self.assertEqual(int(split.optimizer.state["step"]), generation + 1)

    def test_parameters_stay_fixed_until_all_microbatches_complete(self):
        trainer = self.trainer()
        before = {
            key: np.array(value)
            for key, value in tree_flatten(trainer.model.parameters())
        }

        def batches():
            for batch in self.microbatches():
                self.assert_trees_close(before, trainer.model.parameters())
                self.assertEqual(int(trainer.optimizer.state["step"]), 0)
                yield batch
            self.assert_trees_close(before, trainer.model.parameters())

        trainer.train_on_batches(batches())
        self.assertEqual(int(trainer.optimizer.state["step"]), 1)

    def test_training_loop_applies_one_update_for_the_generation(self):
        trainer = self.trainer()
        with (
            patch("model.training.generate_games", return_value=[]),
            patch("model.training.create_batches", return_value=self.microbatches()),
        ):
            trainer.train(n_epochs=1)
        self.assertEqual(int(trainer.optimizer.state["step"]), 1)

    def test_empty_generation_does_not_update(self):
        trainer = self.trainer()
        self.assertEqual(trainer.train_on_batches([]), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(int(trainer.optimizer.state["step"]), 0)

    def test_failed_generation_does_not_apply_partial_update(self):
        trainer = self.trainer()
        before = trainer.model.parameters()

        def failing_batches():
            yield self.batch
            raise RuntimeError("batch failed")

        with (
            self.assertLogs("model.training", level="ERROR"),
            self.assertRaisesRegex(RuntimeError, "batch failed"),
        ):
            trainer.train_on_batches(failing_batches())
        self.assertEqual(int(trainer.optimizer.state["step"]), 0)
        self.assert_trees_close(before, trainer.model.parameters())

    def test_batchnorm_state_is_updated_and_excluded_from_adam(self):
        class BatchNormNet(TinyNet):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm(2)

            def forward_logits(self, states):
                return super().forward_logits(self.bn(states))

        model = BatchNormNet()
        reference = BatchNormNet()
        reference.load_weights(tree_flatten(model.parameters()))
        trainer = self.trainer(model)
        for batch in self.microbatches():
            reference.forward_logits(batch[0])
            mx.eval(reference.parameters())
        trainer.train_on_batches(self.microbatches())
        np.testing.assert_allclose(model.bn.running_mean, reference.bn.running_mean)
        np.testing.assert_allclose(model.bn.running_var, reference.bn.running_var)
        optimizer_keys = dict(tree_flatten(trainer.optimizer.state))
        self.assertFalse(
            any("running_mean" in key or "running_var" in key for key in optimizer_keys)
        )

    def test_tail_batch_contains_each_position_once_in_order(self):
        history = []
        for i in range(5):
            state = np.zeros((19, 8, 8), dtype=np.float32)
            state[0, 0, 0] = i
            state[12].fill(1 - i % 2)
            history.append((mx.array(state), mx.ones(3) / 3, None))
        batches = list(create_batches([(history, 1.0)], batch_size=3))
        self.assertEqual([len(batch[0]) for batch in batches], [3, 2])
        positions = mx.concatenate([batch[0][:, 0, 0, 0] for batch in batches])
        np.testing.assert_array_equal(positions, range(5))


if __name__ == "__main__":
    unittest.main()
