import unittest
import mlx.core as mx
import numpy as np
from model.inference import BatchedInference
from test_shared_inference import small_model


class TestServerPredictionCache(unittest.TestCase):
    def test_deduplication_lru_model_isolation_and_new_snapshot(self):
        first, config = small_model()
        second, _ = small_model(2)
        config.prediction_cache_size = 1
        state = mx.ones((1, 19, 8, 8))
        with BatchedInference(first, config, reference_model=second) as service:
            a, b = service.client(), service.reference_client()
            expected = np.array(a(state)[0])
            np.testing.assert_array_equal(a(state)[0], expected)
            self.assertEqual(service.rpc.statistics()["positions"], 1)
            self.assertEqual(service.rpc.statistics()["cache_hits"], 1)
            self.assertFalse(np.array_equal(b(state)[0], expected))
            a(state)  # Reference weights evicted the current entry.
            self.assertEqual(service.rpc.statistics()["positions"], 3)
        first.policy_fc.bias = mx.array([3.0, 0.0, 0.0, 0.0])
        with BatchedInference(first, config) as fresh:
            self.assertFalse(np.array_equal(fresh.client()(state)[0], expected))

    def test_cache_hits_and_duplicates_do_not_consume_gpu_batch_slots(self):
        from concurrent.futures import Future
        from queue import Queue
        from unittest.mock import Mock
        from cachetools import LRUCache
        from model.inference_backend import _InferenceBackend

        backend = _InferenceBackend.__new__(_InferenceBackend)
        backend.device = mx.cpu
        backend.batch_size, backend.cache_hits, backend.error = 2, 0, None
        backend.requests, backend.predictions = Queue(), LRUCache(maxsize=1)
        cached = np.full((19, 8, 8), 9, dtype=np.float32)
        expected = (np.array([[0.25] * 4]), np.array([[0.0]]))
        backend.predictions[("current", cached.shape, cached.tobytes())] = expected
        responses = [Future() for _ in range(4)]
        for position, response in zip([np.zeros_like(cached), cached,
                                       np.zeros_like(cached), np.ones_like(cached)], responses):
            backend.requests.put(("current", position, response))
        backend.requests.put(None)

        def forward(pending):
            self.assertTrue(responses[1].done())  # Hit returned before GPU work.
            self.assertEqual(len(pending), 2)  # Both miss slots were filled.
            self.assertEqual(sorted(len(waiters) for _, waiters in pending.values()), [1, 2])
            for _, waiters in pending.values():
                for response in waiters:
                    response.set_result(expected)

        backend._predict = Mock(side_effect=forward)
        backend._serve()
        backend._predict.assert_called_once()
        self.assertEqual(backend.cache_hits, 2)
        self.assertTrue(all(response.done() for response in responses))
