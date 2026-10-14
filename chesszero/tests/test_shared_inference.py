import multiprocessing as mp
import os
import sys
import unittest

import numpy as np

from config.model_config import ModelConfig
from model.game_workers import run_game_workers
from model.inference import BatchedInference, InferenceClient


def small_model(bias=0):
    import mlx.core as mx
    from model.network import ChessNet

    config = ModelConfig(
        n_residual_blocks=0, n_filters=2, policy_output_dim=4, inference_batch_size=2
    )
    model = ChessNet(config)
    model.policy_fc.weight = mx.zeros_like(model.policy_fc.weight)
    model.policy_fc.bias = mx.array([bias, 0.0, 0.0, 0.0])
    model.unpicklable = lambda: None  # Only weights go to the server, never the model.
    return model, config


def prediction_worker(client, index, barrier, sender, max_workers):
    assert "mlx.core" not in sys.modules
    assert isinstance(client, InferenceClient)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    if barrier is not None:
        barrier.wait(timeout=10)
    try:
        policy, _ = client(np.full((1, 19, 8, 8), index, dtype=np.float32))
        sender.send((np.array(policy), "cpu"))
    finally:
        client.close()


class TestSharedInference(unittest.TestCase):
    def test_spawned_games_use_one_process_with_bounded_queued_inference(self):
        model, config = small_model()
        barrier = mp.get_context("spawn").Barrier(4)
        with BatchedInference(model, config) as service:
            self.assertNotEqual(service.manager._process.pid, os.getpid())
            self.assertIsInstance(service.manager.address, str)  # Local IPC, not TCP.
            results, failures = run_game_workers(
                prediction_worker,
                [(i, barrier) for i in range(4)],
                4,
                (service.client(),),
            )
        self.assertEqual(failures, 0)
        self.assertEqual(service.statistics["positions"], 4)
        self.assertLessEqual(service.statistics["largest_batch"], 2)
        for policy, device in results:
            np.testing.assert_allclose(policy, [[0.25] * 4])
            self.assertIn("cpu", device)

    def test_rpc_failure_returns_to_caller_and_shutdown_completes(self):
        model, config = small_model()
        with self.assertRaisesRegex(RuntimeError, "Shared inference failed"):
            with BatchedInference(model, config) as service:
                client = service.client()
                try:
                    with self.assertRaises(RuntimeError):
                        client(np.zeros((1, 1), dtype=np.float32))  # Invalid chess input fails remotely.
                finally:
                    client.close()

    def test_evaluation_checkpoints_share_the_same_process_and_model_architecture(self):
        current, config = small_model(0)
        reference, _ = small_model(2)
        with BatchedInference(current, config, reference_model=reference) as service:
            a, b = service.client(), service.reference_client()
            try:
                state = np.ones((1, 19, 8, 8), dtype=np.float32)
                np.testing.assert_allclose(a(state)[0], [[0.25] * 4])
                expected = np.exp([[2.0, 0.0, 0.0, 0.0]]) / (np.exp(2) + 3)
                np.testing.assert_allclose(b(state)[0], expected, rtol=1e-6)
                np.testing.assert_allclose(a(state)[0], [[0.25] * 4])
            finally:
                a.close()
                b.close()
