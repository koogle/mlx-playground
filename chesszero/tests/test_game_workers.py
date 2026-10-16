import os
import tempfile
import unittest
from multiprocessing.reduction import ForkingPickler
from unittest.mock import Mock, patch

import mlx.core as mx
import numpy as np

from chess_engine.bitboard import BitBoard
from config.model_config import ModelConfig
from types import SimpleNamespace
from model.evaluation import play_evaluation_game, play_promotion_game
from model.game_workers import run_game_workers
from model.mcts import MCTS, Node
from model.self_play import generate_games
from model.training import IncompleteEvaluationError, Trainer


def worker_probe(kind, sender, max_workers):
    if kind == "crash":
        os._exit(7)
    if kind != "missing":
        sender.send(None if kind == "failure" else kind)


def worker_limit_probe(sender, max_workers):
    sender.send(max_workers)


class TestGameWorkers(unittest.TestCase):
    def test_cuda_games_are_serialized_without_dropping_jobs(self):
        with (
            patch(
                "model.game_workers.mx.cuda",
                SimpleNamespace(is_available=lambda: True),
                create=True,
            ),
            patch("model.game_workers.mx.default_device", return_value=mx.gpu),
            self.assertLogs("model.game_workers", level="WARNING") as logs,
        ):
            results, failures = run_game_workers(worker_limit_probe, [()] * 3, 512)
        self.assertEqual(results, [1, 1, 1])
        self.assertEqual(failures, 0)
        self.assertIn("from 512 to 1", logs.output[0])

    def test_cpu_and_metal_keep_requested_parallelism(self):
        for cuda_available, device in ((True, mx.cpu), (False, mx.gpu)):
            with (
                self.subTest(cuda_available=cuda_available, device=device),
                patch(
                    "model.game_workers.mx.cuda",
                    SimpleNamespace(is_available=lambda: cuda_available),
                    create=True,
                ),
                patch("model.game_workers.mx.default_device", return_value=device),
            ):
                results, failures = run_game_workers(worker_limit_probe, [()] * 3, 2)
            self.assertEqual(results, [2, 2, 2])
            self.assertEqual(failures, 0)

    def test_spawn_counts_reported_failures_crashes_and_missing_results(self):
        jobs = [(kind,) for kind in ("first", "failure", "crash", "missing", "last")]
        results, failures = run_game_workers(worker_probe, jobs, max_workers=2)
        self.assertEqual(results, ["first", "last"])
        self.assertEqual(failures, 3)

    def test_evaluation_target_is_picklable_and_failed_matches_are_not_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(ModelConfig(n_residual_blocks=0, n_filters=2), directory)
            ForkingPickler.dumps(play_evaluation_game)
            trainer._accept_current_model()
            with patch(
                "model.training.run_pipe_games", return_value=([{"result": 1}], 1, {})
            ):
                with self.assertRaises(IncompleteEvaluationError):
                    trainer.evaluate_against_best(n_games=2)

    def test_all_failed_self_play_cannot_produce_an_update(self):
        with patch("model.self_play.run_pipe_games", return_value=([], 2, {})):
            with self.assertRaisesRegex(RuntimeError, "Incomplete self-play"):
                generate_games(None, ModelConfig(n_games_per_iteration=2))

    def test_partial_self_play_is_rejected(self):
        with patch("model.self_play.run_pipe_games", return_value=([{}], 1, {})):
            with self.assertRaisesRegex(RuntimeError, "Incomplete self-play"):
                generate_games(None, ModelConfig(n_games_per_iteration=2))

    def test_random_evaluation_scores_draws_and_requires_every_game(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(ModelConfig(n_residual_blocks=0, n_filters=2), directory)
            results = [{"result": r} for r in (1, -1, 0, 0)]
            with patch(
                "model.training.run_pipe_games", return_value=(results, 0, {})
            ) as run:
                self.assertEqual(trainer.evaluate_against_random(4, 2), (0.5, 1, 1, 2))
                jobs = run.call_args.args[3]
                self.assertEqual([job[2] for job in jobs], [0, 1, 0, 1])
            with patch(
                "model.training.run_pipe_games", return_value=(results[:2], 0, {})
            ):
                with self.assertRaises(IncompleteEvaluationError):
                    trainer.evaluate_against_random(4, 2)

    def test_promotion_openings_are_legal_paired_and_seeded(self):
        import chess
        from chess_engine.game import ChessGame
        from test_board import coords, from_fen

        class StopAtOpening(Exception):
            pass

        openings = []
        for seed in (20000, 20001):
            pair = []
            for color in (0, 1):
                game, oracle = ChessGame(), chess.Board()
                original_make_move = game.make_move

                def apply(source, target):
                    legal = {coords(move): move for move in oracle.legal_moves}
                    self.assertIn((source, target), legal)
                    oracle.push(legal[(source, target)])
                    return original_make_move(source, target)

                def capture(board, temperature):
                    self.assertEqual(board.get_move_count(), 8)
                    expected, _ = from_fen(oracle.fen(en_passant="fen"))
                    np.testing.assert_array_equal(board.state, expected.state)
                    pair.append(board.get_hash())
                    raise StopAtOpening

                client, connection = Mock(), Mock()
                with (
                    patch("model.evaluation.ChessGame", return_value=game),
                    patch.object(game, "make_move", side_effect=apply),
                    patch("model.evaluation.MCTS") as search,
                ):
                    search.return_value.get_move.side_effect = capture
                    with self.assertRaises(StopAtOpening):
                        play_promotion_game(
                            client, ModelConfig(), 0, color, False, seed, connection, 1
                        )
                client.close.assert_called_once()
            self.assertEqual(pair[0], pair[1])
            openings.append(pair[0])
        self.assertNotEqual(*openings)

    def test_promotion_jobs_balance_colors_on_identical_opening_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(ModelConfig(n_residual_blocks=0, n_filters=2), directory)
            trainer._accept_current_model()
            results = [{"result": r} for r in (1, 0, 0, -1)]
            with patch(
                "model.training.run_pipe_games", return_value=(results, 0, {})
            ) as run:
                self.assertEqual(
                    trainer.evaluate_against_best(4, 2, seed=123), (0.5, 1, 1, 2)
                )
                self.assertIs(run.call_args.args[2], play_promotion_game)
                jobs = run.call_args.args[3]
                self.assertEqual(
                    [(j[2], j[4]) for j in jobs],
                    [(0, 123), (0, 124), (1, 123), (1, 124)],
                )

    def test_configured_root_noise_affects_selection_without_changing_priors(self):
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
            priors = search._root_search_priors(root)
            np.testing.assert_allclose(priors, expected)
            self.assertIs(root.select_child(1.0, priors)[1], children[0])
            children[0].visit_count = 1
            self.assertIs(root.select_child(1.0, priors)[1], children[0])
            np.testing.assert_allclose([c.prior for c in children], np.full(20, 1 / 20))
            sample.assert_called_once_with([0.7] * 20)
