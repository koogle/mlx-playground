import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from config.model_config import ModelConfig
from model.inference_mode import search_inference_mode
from model.network import ChessNet
from model.search_batch_norm import SearchBatchNorm


def test_independent_search_matches_serial_bn_and_ignores_other_positions():
    mx.random.seed(94)
    model = ChessNet(ModelConfig(n_filters=4, n_residual_blocks=1, policy_output_dim=8))
    states = mx.random.normal((3, 19, 8, 8))
    before = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
    with search_inference_mode(model, True):
        serial = [model(states[i : i + 1]) for i in range(3)]
        mixed = model(states)
    # Demonstrate that the prior batch-stat implementation changes predictions.
    assert np.max(np.abs(np.asarray(mixed[0][0]) - np.asarray(serial[0][0][0]))) > 1e-6
    with search_inference_mode(model, True, per_position=True):
        parallel = model(states)
        reordered = model(states[mx.array([2, 0, 1])])
        for head in (0, 1):
            expected = np.concatenate([np.asarray(s[head]) for s in serial])
            np.testing.assert_allclose(parallel[head], expected, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(
                reordered[head], expected[[2, 0, 1]], rtol=1e-5, atol=1e-6
            )
    for key, value in tree_flatten(model.parameters()):
        np.testing.assert_array_equal(value, before[key])


def test_training_matches_standard_batchnorm_outputs_and_running_statistics():
    standard, compatible = nn.BatchNorm(3), SearchBatchNorm(3)
    for offset in (0, 2, -1):
        states = mx.arange(96, dtype=mx.float32).reshape(2, 4, 4, 3) + offset
        np.testing.assert_array_equal(standard(states), compatible(states))
        np.testing.assert_array_equal(standard.running_mean, compatible.running_mean)
        np.testing.assert_array_equal(standard.running_var, compatible.running_var)


def test_search_position_mode_restored_on_exception():
    model = ChessNet(ModelConfig(n_filters=2, n_residual_blocks=1))
    layers = [m for m in model.modules() if isinstance(m, SearchBatchNorm)]
    with pytest.raises(RuntimeError):
        with search_inference_mode(model, True, per_position=True):
            assert all(m._search_per_position for m in layers)
            raise RuntimeError("prediction failed")
    assert model.training
    assert all(not m._search_per_position and m.track_running_stats for m in layers)
