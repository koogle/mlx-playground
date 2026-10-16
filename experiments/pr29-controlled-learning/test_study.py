import importlib.util
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def load_study():
    spec = importlib.util.spec_from_file_location(
        "controlled_study", Path(__file__).with_name("study.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = nn.Linear(2, 3)
        self.value = nn.Linear(2, 1)

    def forward_logits(self, states):
        return self.policy(states), self.value(states)


def test_disabled_value_loss_does_not_depend_on_value_labels():
    study = load_study()
    model = TinyModel()
    states, policies = mx.ones((2, 2)), mx.ones((2, 3)) / 3
    first = study.objective(model, states, policies, mx.ones(2), "policy_only")[0]
    second = study.objective(model, states, policies, -mx.ones(2), "policy_only")[0]
    np.testing.assert_allclose(first, second)


def test_joint_objective_matches_production_formula():
    study = load_study()
    from model.training import Trainer

    model = TinyModel()
    inputs = mx.ones((2, 2)), mx.ones((2, 3)) / 3, mx.ones(2)
    expected = Trainer._loss_fn(type("Owner", (), {"model": model})(), *inputs)
    actual = study.objective(model, *inputs, "joint_ordered")
    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])


def test_shuffling_preserves_sample_policy_and_label_alignment():
    study = load_study()
    ids = np.arange(133, dtype=np.float32)
    arrays = ids[:, None], (ids + 10)[:, None], ids + 20
    batches = list(study.batches(arrays, shuffled=True))
    states, policies, targets = (
        np.concatenate([np.asarray(b[i]) for b in batches]) for i in range(3)
    )
    np.testing.assert_array_equal(np.sort(states[:, 0]), ids)
    np.testing.assert_array_equal(policies[:, 0], states[:, 0] + 10)
    np.testing.assert_array_equal(targets, states[:, 0] + 20)
    assert not np.array_equal(states[:, 0], ids)
    assert [len(b[0]) for b in batches] == [64, 64, 5]
