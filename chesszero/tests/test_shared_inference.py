import multiprocessing as mp
import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from config.model_config import ModelConfig
from model.game_workers import run_game_workers
from model.inference import BatchedInference, InferenceClient


class AnalyticModel(nn.Module):
    def __init__(self, offset=0, fail=False):
        super().__init__()
        self.offset, self.fail = offset, fail
        # The model must never be serialized into the game workers.
        self.unpicklable = lambda: None

    def __call__(self, states):
        if self.fail:
            raise RuntimeError("broken forward")
        values = states[:, 0, 0, 0] + self.offset
        return mx.stack([values, -values], axis=1), values[:, None]


def prediction_worker(first, second, index, barrier, sender, max_workers):
    mx.set_default_device(mx.cpu)
    assert isinstance(first, InferenceClient)
    if barrier is not None:
        barrier.wait(timeout=10)
    state = mx.full((1, 19, 8, 8), index)
    try:
        a, _ = first(state)
        b, _ = second(state)
        sender.send((np.array(a), np.array(b), str(mx.default_device())))
    except RuntimeError as error:
        sender.send(str(error))


class TestSharedInference(unittest.TestCase):
    def test_spawned_games_share_models_batch_and_keep_outputs_separate(self):
        first, second = AnalyticModel(), AnalyticModel(offset=10)
        config = ModelConfig(inference_batch_size=4, inference_wait_ms=50)
        barrier = mp.get_context("spawn").Barrier(4)
        jobs = [(first, second, i, barrier) for i in range(4)]
        with BatchedInference([first, second], config) as service:
            results, failures = run_game_workers(
                prediction_worker, jobs, 4, inference=service
            )
            self.assertFalse(first.training)
            self.assertEqual(failures, 0)
        self.assertTrue(first.training)
        self.assertEqual(service.positions, 8)
        self.assertEqual(service.max_batch_size, 4)
        self.assertEqual(service.forward_passes, 2)
        for i, (a, b, device) in enumerate(results):
            np.testing.assert_array_equal(a, [[i, -i]])
            np.testing.assert_array_equal(b, [[i + 10, -i - 10]])
            self.assertIn("cpu", device)

    def test_forward_failure_disconnects_waiter_and_restores_mode(self):
        model = AnalyticModel(fail=True)
        with self.assertRaisesRegex(RuntimeError, "Shared inference failed"):
            with BatchedInference([model], ModelConfig()) as service:
                results, failures = run_game_workers(
                    prediction_worker, [(model, model, 0, None)], 1, inference=service
                )
        self.assertTrue(model.training)
        self.assertEqual(failures, 0)
        self.assertIn("disconnected", results[0])

    def test_multi_position_requests_respect_forward_batch_limit(self):
        model = AnalyticModel()
        with BatchedInference([model], ModelConfig(inference_batch_size=2)) as service:
            args, connection = service.prepare_job((model,), mp.get_context("spawn"))
            try:
                policies, values = args[0](
                    mx.broadcast_to(mx.arange(5)[:, None, None, None], (5, 19, 8, 8))
                )
            finally:
                connection.close()
        np.testing.assert_array_equal(values, np.arange(5)[:, None])
        np.testing.assert_array_equal(
            policies, np.stack([np.arange(5), -np.arange(5)], axis=1)
        )
        self.assertEqual(service.max_batch_size, 2)
        self.assertEqual(service.positions, 5)
