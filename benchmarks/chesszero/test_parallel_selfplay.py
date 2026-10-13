"""Snapshot isolation, failure propagation, and non-mutating learning diagnostics."""

from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import parallel_selfplay
from convergence import value_mode_probe, training_games


@dataclass
class Config:
    n_games_per_iteration: int = 2


class Model:
    value = "initial"

    def save_weights(self, path):
        Path(path).write_text(self.value)


class Pool:
    def __init__(self, **kwargs):
        self.futures = []
        self.args = []

    def submit(self, function, *args):
        future = Future()
        self.futures.append(future)
        self.args.append(args)
        return future

    def shutdown(self, **kwargs):
        pass


class PipelineTests(unittest.TestCase):
    def test_snapshot_isolated_and_games_consumed_once_in_order(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(parallel_selfplay, "ProcessPoolExecutor", Pool),
        ):
            pipeline = parallel_selfplay.SelfPlayPipeline("repo", Config(), tmp, 2)
            model = Model()
            pipeline.submit(model, 2, 0)
            model.value = "learned"
            self.assertEqual(Path(pipeline.pool.args[0][2]).read_text(), "initial")
            self.assertNotEqual(pipeline.pool.args[0][-1], pipeline.pool.args[1][-1])
            with self.assertRaises(RuntimeError):
                pipeline.submit(model, 3, 1)
            pipeline.pool.futures[1].set_result("second")
            pipeline.pool.futures[0].set_result("first")
            games, version, _ = pipeline.collect()
            self.assertEqual(games, ["first", "second"])
            self.assertEqual(version, 0)
            self.assertIsNone(pipeline.pending)
            self.assertFalse(list(Path(tmp).glob("*.safetensors")))

    def test_worker_failure_propagates(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(parallel_selfplay, "ProcessPoolExecutor", Pool),
        ):
            pipeline = parallel_selfplay.SelfPlayPipeline("repo", Config(), tmp, 2)
            pipeline.submit(Model(), 1, 0)
            pipeline.pool.futures[0].set_exception(RuntimeError("worker failed"))
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                pipeline.collect()

    def test_cpu_worker_records_train_through_production_batcher(self):
        from chess_engine.bitboard import BitBoard
        from config.model_config import ModelConfig
        from model.self_play import create_batches
        from model.training import Trainer

        state = np.array(BitBoard().state)
        policy = np.zeros(4672, dtype=np.float32)
        policy[0] = 1
        games = [([(state, policy, None)] * 3, 1.0)]
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(ModelConfig(n_residual_blocks=1, n_filters=8), tmp)
            losses = trainer.train_on_batches(create_batches(training_games(games), 2))
            self.assertEqual(int(trainer.optimizer.state["step"]), 1)
            self.assertTrue(all(np.isfinite(v) for v in losses))

    def test_value_probe_preserves_all_parameters_and_running_statistics(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from config.model_config import ModelConfig
        from model.network import ChessNet
        from chess_engine.bitboard import BitBoard

        model = ChessNet(ModelConfig(n_residual_blocks=1, n_filters=8))
        mx.eval(model.parameters())
        before = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
        state = np.array(BitBoard().state)
        metrics = value_mode_probe(model, [([(state, None, None)] * 2, 1.0)])
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        for key, value in tree_flatten(model.parameters()):
            np.testing.assert_array_equal(before[key], np.array(value))


if __name__ == "__main__":
    unittest.main()
