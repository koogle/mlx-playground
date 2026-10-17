import sys
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from study import objective


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = mx.array([1.0, 2.0, 80.0, -1.0])

    def forward_logits(self, states):
        return self.logits[None, :], mx.zeros((1, 1))


def test_legal_loss_and_gradient_ignore_illegal_logits():
    model = Toy()
    states = mx.zeros((1, 1))
    targets = mx.array([[0.25, 0.75, 0.0, 0.0]])
    mask = mx.array([[True, True, False, False]])
    loss = lambda m: objective(m, states, targets, mx.zeros(1), "legal_policy", mask)[0]
    value, grads = nn.value_and_grad(model, loss)(model)
    expected = nn.losses.cross_entropy(mx.array([[1.0, 2.0]]), mx.array([[0.25, 0.75]]))
    np.testing.assert_allclose(
        np.asarray(value), np.asarray(expected).mean(), atol=1e-6
    )
    gradient = np.asarray(grads["logits"])
    assert np.isfinite(gradient).all()
    assert np.linalg.norm(gradient[:2]) > 0
    np.testing.assert_array_equal(gradient[2:], 0)
    model.logits = mx.array([1.0, 2.0, -900.0, 900.0])
    np.testing.assert_allclose(np.asarray(loss(model)), np.asarray(value), atol=1e-6)
