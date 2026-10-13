import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from config.model_config import ModelConfig
from model.network import ChessNet
from model.training import Trainer
from model.mcts import Node
from chess_engine.bitboard import BitBoard


class TestCheckpoints(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = ModelConfig(
            n_filters=2, n_residual_blocks=0, policy_output_dim=4, learning_rate=1e-3
        )
        self.batch = (
            mx.ones((2, 19, 8, 8)),
            mx.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]),
            mx.array([1.0, -1.0]),
        )

    def trainer(self, **kwargs):
        return Trainer(self.config, self.directory.name, **kwargs)

    def assert_trees_equal(self, left, right):
        left, right = dict(tree_flatten(left)), dict(tree_flatten(right))
        self.assertEqual(left.keys(), right.keys())
        for key in left:
            np.testing.assert_allclose(
                left[key], right[key], atol=1e-7, rtol=1e-6, err_msg=key
            )

    def test_resumed_next_update_matches_uninterrupted_model_and_adam(self):
        original = self.trainer()
        original.train_on_batch(self.batch)
        original._save_checkpoint(1)
        resumed = self.trainer(resume_epoch=1)
        self.assertEqual(resumed.start_epoch, 1)
        self.assert_trees_equal(original.optimizer.state, resumed.optimizer.state)
        original.train_on_batch(self.batch)
        resumed.train_on_batch(self.batch)
        self.assert_trees_equal(original.model.parameters(), resumed.model.parameters())
        self.assert_trees_equal(original.optimizer.state, resumed.optimizer.state)

    def test_rejection_restores_weights_moments_and_step(self):
        trainer = self.trainer()
        trainer.train_on_batch(self.batch)
        trainer._accept_current_model(0.0)
        accepted_model = trainer.best_model.parameters()
        accepted_optimizer = trainer.best_optimizer_state
        trainer.train_on_batch(self.batch)
        self.assertEqual(int(trainer.optimizer.state["step"]), 2)
        self.assertEqual(int(accepted_optimizer["step"]), 1)
        trainer._revert_to_best()
        self.assert_trees_equal(trainer.model.parameters(), accepted_model)
        self.assert_trees_equal(trainer.optimizer.state, accepted_optimizer)

    def test_interrupted_and_regular_snapshots_are_separate_and_keep_best(self):
        trainer = self.trainer()
        trainer.train_on_batch(self.batch)
        trainer._accept_current_model(0.0)
        trainer._save_checkpoint(1)
        trainer.train_on_batch(self.batch)
        trainer._save_checkpoint(1, interrupted=True, completed_epochs=0)
        regular = self.trainer(resume_epoch=1)
        interrupted = self.trainer(resume_epoch=1, resume_interrupted=True)
        self.assertEqual(regular.start_epoch, 1)
        self.assertEqual(interrupted.start_epoch, 0)
        self.assertEqual(int(regular.optimizer.state["step"]), 1)
        self.assertEqual(int(interrupted.optimizer.state["step"]), 2)
        self.assertIsNotNone(regular.best_model)  # A zero score is still a best model.
        self.assert_trees_equal(
            interrupted.best_model.parameters(), regular.model.parameters()
        )
        interrupted._revert_to_best()
        self.assert_trees_equal(interrupted.optimizer.state, regular.optimizer.state)

    def test_failed_save_keeps_previous_manifest_usable(self):
        trainer = self.trainer()
        trainer._save_checkpoint(1)
        manifest = Path(self.directory.name, "state_epoch_1.json").read_text()
        with patch(
            "model.network.mx.save_safetensors", side_effect=RuntimeError("disk failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "disk failed"):
                trainer._save_checkpoint(1)
        self.assertEqual(
            Path(self.directory.name, "state_epoch_1.json").read_text(), manifest
        )
        self.trainer(resume_epoch=1)

    def test_legacy_optimizer_restarts_at_zero_with_saved_learning_rate(self):
        trainer = self.trainer()
        trainer.model.save_weights(
            str(Path(self.directory.name, "model_epoch_5.safetensors"))
        )
        Path(self.directory.name, "state_epoch_5.json").write_text(
            json.dumps(
                {
                    "epoch": 5,
                    "config": vars(self.config),
                    "best_model_win_rate": 0.0,
                    "optimizer_state": {"step": 300, "learning_rate": 0.0003},
                }
            )
        )
        with self.assertWarnsRegex(UserWarning, "lacks Adam moments"):
            resumed = self.trainer(resume_epoch=5)
        self.assertEqual(int(resumed.optimizer.state["step"]), 0)
        self.assertAlmostEqual(float(resumed.optimizer.learning_rate), 0.0003)
        self.assertEqual(resumed.start_epoch, 5)
        self.assertIsNotNone(resumed.best_model)

    def test_draws_contribute_half_a_point(self):
        self.assertEqual(Trainer._evaluation_score(0, 0, 40), 0.5)
        self.assertEqual(Trainer._evaluation_score(1, 0, 39), 0.5125)
        self.assertEqual(Trainer._evaluation_score(24, 16, 0), 0.6)
        with self.assertRaises(ValueError):
            Trainer._evaluation_score(0, 0, 0)

    def test_evaluation_selection_never_draws_search_noise(self):
        root = Node(BitBoard(), exploration_noise=False)
        root.children = {i: Node(root.board, parent=root, prior=0.5) for i in range(2)}
        with (
            patch("numpy.random.normal", side_effect=AssertionError("noise")),
            patch("numpy.random.dirichlet", side_effect=AssertionError("noise")),
        ):
            root.select_child(1.0)
            root.children[0].visit_count = 1
            root.select_child(1.0)


if __name__ == "__main__":
    unittest.main()
