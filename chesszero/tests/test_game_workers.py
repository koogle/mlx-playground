import os
import tempfile
import unittest
from multiprocessing.reduction import ForkingPickler
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from chess_engine.bitboard import BitBoard
from config.model_config import ModelConfig
from model.game_workers import run_game_workers
from model.mcts import MCTS, Node
from model.self_play import generate_games
from model.training import IncompleteEvaluationError, Trainer


def worker_probe(kind, sender, max_workers):
    if kind == "crash":
        os._exit(7)
    if kind != "missing":
        sender.send(None if kind == "failure" else kind)


class TestGameWorkers(unittest.TestCase):
    def test_spawn_counts_reported_failures_crashes_and_missing_results(self):
        jobs = [(kind,) for kind in ("first", "failure", "crash", "missing", "last")]
        results, failures = run_game_workers(worker_probe, jobs, max_workers=2)
        self.assertEqual(results, ["first", "last"])
        self.assertEqual(failures, 3)

    def test_evaluation_target_is_picklable_and_failed_matches_are_not_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(ModelConfig(n_residual_blocks=0, n_filters=2), directory)
            ForkingPickler.dumps(trainer.play_evaluation_game_worker)
            trainer._accept_current_model()
            with patch(
                "model.training.run_game_workers", return_value=([{"result": 1}], 1)
            ):
                with self.assertRaises(IncompleteEvaluationError):
                    trainer.evaluate_against_best(n_games=2)

    def test_all_failed_self_play_cannot_produce_an_update(self):
        with patch("model.self_play.run_game_workers", return_value=([], 2)):
            with self.assertRaisesRegex(RuntimeError, "All self-play games failed"):
                generate_games(None, ModelConfig(n_games_per_iteration=2))

    def test_configured_root_noise_persists_in_priors(self):
        config = ModelConfig(dirichlet_alpha=0.7, dirichlet_epsilon=0.4)
        search = MCTS(None, config)
        root = Node(BitBoard())
        noise = np.zeros(20)
        noise[0] = 1
        with patch("numpy.random.dirichlet", return_value=noise) as sample:
            search._expand_node(root, mx.ones(4672) / 4672, 0.0)
            children = list(root.children.values())
            expected = np.full(20, 0.6 / 20)
            expected[0] += 0.4
            np.testing.assert_allclose([c.prior for c in children], expected)
            children[0].visit_count = 1
            root.select_child(1.0)
            np.testing.assert_allclose([c.prior for c in children], expected)
            sample.assert_called_once_with([0.7] * 20)
