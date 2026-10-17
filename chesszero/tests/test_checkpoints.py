import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from config.model_config import ModelConfig
from model.training import Trainer, IncompleteEvaluationError
from model.mcts import Node
from chess_engine.bitboard import BitBoard


class TestCheckpoints(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = ModelConfig(
            n_filters=2, n_residual_blocks=1, policy_output_dim=4, learning_rate=1e-3
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

    def test_loop_promotes_winner_then_rejects_and_resumes_full_teacher(self):
        trainer = self.trainer()
        candidates, teachers = [], []

        def generate(teacher, *args):
            self.assertIs(teacher, trainer.best_model)
            self.assertIsNot(teacher, trainer.model)
            teachers.append(teacher.parameters())
            return []

        def evaluate(**kwargs):
            self.assertEqual(
                int(trainer.optimizer.state["step"]), 2 * (len(candidates) + 1)
            )
            candidates.append(trainer.model.parameters())
            return (0.6, 24, 16, 0) if len(candidates) == 1 else (0.5, 0, 0, 40)

        with (
            patch("model.training.generate_games", side_effect=generate),
            patch(
                "model.training.create_batches",
                side_effect=lambda *args: iter([self.batch] * 2),
            ),
            patch.object(trainer, "evaluate_against_best", side_effect=evaluate),
        ):
            trainer.train(n_epochs=2, n_workers=2)
        self.assert_trees_equal(teachers[1], candidates[0])
        self.assert_trees_equal(trainer.model.parameters(), candidates[0])
        self.assertEqual(int(trainer.optimizer.state["step"]), 2)
        resumed = self.trainer(resume_epoch=2)
        self.assertEqual(resumed.start_epoch, 2)
        self.assert_trees_equal(resumed.model.parameters(), candidates[0])
        self.assert_trees_equal(resumed.best_model.parameters(), candidates[0])
        self.assert_trees_equal(resumed.optimizer.state, trainer.optimizer.state)
        self.assert_trees_equal(resumed.best_optimizer_state, trainer.optimizer.state)

    def test_incomplete_first_match_reverts_to_initial_checkpoint(self):
        trainer = self.trainer()
        initial = trainer.model.parameters()
        with (
            patch("model.training.generate_games", return_value=[]),
            patch("model.training.create_batches", return_value=iter([self.batch])),
            patch.object(
                trainer,
                "evaluate_against_best",
                side_effect=IncompleteEvaluationError("one missing game"),
            ),
        ):
            trainer.train(n_epochs=1)
        self.assert_trees_equal(trainer.model.parameters(), initial)
        self.assertEqual(int(trainer.optimizer.state["step"]), 0)
        self.assertEqual(int(self.trainer(resume_epoch=1).optimizer.state["step"]), 0)

    def test_self_play_keeps_accepted_teacher_between_evaluations(self):
        trainer = self.trainer()
        trainer.config.eval_interval_epochs = 2
        initial = trainer.model.parameters()

        def generate(teacher, *args):
            self.assert_trees_equal(teacher.parameters(), initial)
            return []

        with (
            patch("model.training.generate_games", side_effect=generate) as games,
            patch(
                "model.training.create_batches",
                side_effect=lambda *args: iter([self.batch]),
            ),
            patch.object(
                trainer, "evaluate_against_best", return_value=(0.5, 0, 0, 40)
            ) as evaluation,
        ):
            trainer.train(n_epochs=2)
        self.assertEqual(games.call_count, 2)
        self.assertEqual(evaluation.call_count, 1)
        self.assert_trees_equal(trainer.model.parameters(), initial)

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

    def test_promotion_counts_draws_as_half_a_point(self):
        self.assertEqual(Trainer._evaluation_score(0, 0, 40), 0.5)
        self.assertEqual(Trainer._evaluation_score(1, 0, 39), 0.5125)
        self.assertEqual(Trainer._evaluation_score(6, 4, 10), 0.55)
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
