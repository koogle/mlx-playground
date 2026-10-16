import os
import sys
import subprocess
from pathlib import Path
import unittest
import numpy as np
from config.model_config import ModelConfig
from model.pipe_inference import run_pipe_games


class ConstantModel:
    def __init__(self, value):
        self.value = value
        self.training = True
        self.calls = 0

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def __call__(self, states):
        self.calls += 1
        return np.full((len(states), 4), self.value, dtype=np.float32), np.full(
            (len(states), 1), self.value, dtype=np.float32
        )


def request_both(client, connection, workers):
    state = np.zeros((1, 19, 8, 8), dtype=np.float32)
    values = [float(client.for_model(i)(state)[1][0, 0]) for i in (0, 1, 0)]
    connection.send(values)
    client.close()


def crash(client, connection, workers):
    os._exit(2)


def missing(client, connection, workers):
    client.close()


def request_shifted(client, connection, workers):
    state = np.arange(19 * 8 * 8, dtype=np.float32).reshape(1, 19, 8, 8) + 4
    values = [float(client(state)[1][0, 0]) for _ in range(2)]
    connection.send(values)
    client.close()


class TestPipeInference(unittest.TestCase):
    def test_models_are_routed_and_cached_separately_in_parent(self):
        models = [ConstantModel(1), ConstantModel(2)]
        results, failures, stats = run_pipe_games(
            models, ModelConfig(search_batch_norm=False), request_both, [()], 1
        )
        self.assertEqual(results, [[1, 2, 1]])
        self.assertEqual(failures, 0)
        self.assertEqual([m.calls for m in models], [1, 1])
        self.assertTrue(all(m.training for m in models))
        self.assertEqual(stats["cache_hits"], 1)

    def test_batch_norm_inference_uses_batch_stats_without_prediction_cache(self):
        from test_inference_mode import ModeModel

        for batch_stats in (True, False):
            with self.subTest(batch_stats=batch_stats):
                model = ModeModel()
                mean, variance = (
                    np.array(model.bn.running_mean),
                    np.array(model.bn.running_var),
                )
                results, failures, stats = run_pipe_games(
                    model,
                    ModelConfig(search_batch_norm=batch_stats),
                    request_shifted,
                    [()],
                    1,
                )
                self.assertEqual(failures, 0)
                np.testing.assert_allclose(
                    results, [[0, 0]] if batch_stats else [[1, 1]], atol=1e-6
                )
                self.assertEqual(len(model.modes), 2 if batch_stats else 1)
                self.assertEqual(stats["cache_hits"], 0 if batch_stats else 1)
                self.assertTrue(model.training)
                self.assertTrue(model.bn.track_running_stats)
                np.testing.assert_array_equal(model.bn.running_mean, mean)
                np.testing.assert_array_equal(model.bn.running_var, variance)

    def test_pipe_restores_normalization_after_prediction_failure(self):
        from test_inference_mode import ModeModel

        model = ModeModel(fail=True)
        with self.assertRaisesRegex(RuntimeError, "prediction failed"):
            run_pipe_games(model, ModelConfig(), request_shifted, [()], 1)
        self.assertTrue(model.training)
        self.assertTrue(model.bn.track_running_stats)

    def test_spawned_cli_import_does_not_load_mlx(self):
        entrypoint = str(Path(__file__).parents[1] / "train.py")
        script = (
            "import runpy, sys; "
            f"runpy.run_path({entrypoint!r}, run_name='__mp_main__'); "
            "assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)"
        )
        subprocess.run(
            [sys.executable, "-c", script],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
            check=True,
        )

    def test_crash_and_missing_result_are_failures(self):
        for worker in (crash, missing):
            with self.subTest(worker=worker.__name__):
                results, failures, _ = run_pipe_games(
                    ConstantModel(1), ModelConfig(), worker, [()], 1
                )
                self.assertEqual(results, [])
                self.assertEqual(failures, 1)
