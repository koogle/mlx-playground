import importlib.util
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "pr29-first-result"))
sys.path.insert(0, str(Path(__file__).parents[1] / "pr29-random-restart"))
spec = importlib.util.spec_from_file_location(
    "ablate", Path(__file__).with_name("ablate.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Network:
    def __init__(self, policy_offset, value_offset):
        self.policy_offset, self.value_offset = policy_offset, value_offset

    def eval(self):
        pass

    def __call__(self, states):
        base = mx.arange(4672, dtype=mx.float32)[None, :] + self.policy_offset
        return mx.broadcast_to(base, (len(states), 4672)), states[
            :, 12, 0, 0, None
        ] + self.value_offset


@pytest.mark.parametrize(
    "kind,policy_offset,values",
    [
        ("trained_policy_zero_value", 100, [0, 0]),
        ("trained_policy_initial_value", 100, [2, 3]),
        ("initial_policy_trained_value", 10, [5, 6]),
        ("trained_policy_negated_value", 100, [-5, -6]),
        ("mirror_value", 100, [5.5, 5.5]),
    ],
)
def test_interventions_change_only_requested_outputs(kind, policy_offset, values):
    states = np.zeros((2, 19, 8, 8), dtype=np.float32)
    states[1, 12] = 1
    wrapped = module.Intervention(Network(10, 2), Network(100, 5), kind)
    p, v = wrapped(mx.array(states))
    np.testing.assert_array_equal(p[0], np.arange(4672) + policy_offset)
    np.testing.assert_array_equal(np.array(v).reshape(-1), values)


def test_combined_symmetry_outputs_are_equivariant_and_do_not_mutate_inputs():
    states = np.random.default_rng(1).normal(size=(2, 19, 8, 8)).astype(np.float32)
    states[:, 12] = np.arange(2)[:, None, None]
    original = states.copy()
    wrapped = module.Intervention(
        Network(10, 2), Network(100, 5), "mirror_policy_and_value"
    )
    p, v = wrapped(mx.array(states))
    mirrored_p, mirrored_v = wrapped(mx.array(module.mirror_states(states)))
    np.testing.assert_array_equal(
        p, np.array(mirrored_p)[:, module.mirror_policy_indices()]
    )
    np.testing.assert_array_equal(v, mirrored_v)
    np.testing.assert_array_equal(states, original)
    np.testing.assert_array_equal(
        module.mirror_states(module.mirror_states(states)), states
    )
