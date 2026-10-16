import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from config.model_config import ModelConfig
from model.network import ChessNet
from model.inference_mode import search_inference_mode


def test_per_position_training_gradient_ignores_other_batch_positions():
    mx.random.seed(162016)
    model = ChessNet(ModelConfig(n_filters=4, n_residual_blocks=1, policy_output_dim=8))
    x = mx.random.normal((3, 19, 8, 8))

    def loss(net, states):
        logits, value = net.forward_logits(states)
        return nn.losses.cross_entropy(
            logits[:1], mx.array([2]), reduction="mean"
        ) + mx.sum(value[:1] ** 2)

    grad_fn = nn.value_and_grad(model, loss)
    with search_inference_mode(model, True, per_position=True):
        model.train()
        single, gs = grad_fn(model, x[:1])
        mixed, gm = grad_fn(model, x)
        mx.eval(single, mixed, gs, gm)
        np.testing.assert_allclose(single, mixed, atol=1e-5, rtol=1e-5)
        total = 0
        for (ks, vs), (km, vm) in zip(tree_flatten(gs), tree_flatten(gm)):
            assert ks == km
            np.testing.assert_allclose(vs, vm, atol=2e-5, rtol=1e-4)
            total += float(mx.sum(mx.abs(vs)))
        assert total > 0 and np.isfinite(total)
