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
