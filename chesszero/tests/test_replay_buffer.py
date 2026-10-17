import unittest

import mlx.core as mx
import numpy as np

from model.replay_buffer import ReplayBuffer, side_to_move_value


def position(white_to_move, marker=0.0):
    state = np.zeros((19, 8, 8), dtype=np.float32)
    state[12].fill(white_to_move)
    state[0, 0, 0] = marker
    return mx.array(state)


class TestReplayBuffer(unittest.TestCase):
    def test_value_targets_follow_side_to_move(self):
        self.assertEqual(side_to_move_value(position(True), 1.0), 1.0)
        self.assertEqual(side_to_move_value(position(False), 1.0), -1.0)
        self.assertEqual(side_to_move_value(position(False), 0.0), 0.0)

    def test_sparse_policies_round_trip_with_matching_states_and_values(self):
        buffer = ReplayBuffer(capacity=10, policy_output_dim=4, seed=0)
        history = [
            (position(True, 1.0), mx.array([0.75, 0.0, 0.25, 0.0]), None),
            (position(False, 2.0), mx.array([0.0, 0.4, 0.0, 0.6]), None),
        ]
        self.assertEqual(buffer.add([(history, -1.0)]), 2)
        self.assertEqual(len(buffer), 2)
        states, policies, values = buffer.sample(batch_size=50)
        self.assertEqual(states.shape, (2, 19, 8, 8))  # capped at buffer size
        for state, policy, value in zip(
            np.array(states), np.array(policies), np.array(values)
        ):
            if state[0, 0, 0] == 1.0:
                np.testing.assert_allclose(policy, [0.75, 0.0, 0.25, 0.0])
                self.assertEqual(value, -1.0)
            else:
                np.testing.assert_allclose(policy, [0.0, 0.4, 0.0, 0.6])
                self.assertEqual(value, 1.0)

    def test_oldest_generations_are_evicted_when_over_capacity(self):
        buffer = ReplayBuffer(capacity=4, policy_output_dim=2, seed=0)
        for marker in range(1, 4):
            history = [(position(True, marker), mx.array([1.0, 0.0]), None)] * 2
            buffer.add([(history, 0.0)])
        self.assertEqual(len(buffer), 4)
        self.assertEqual(buffer.n_generations, 2)
        markers = {float(s[0, 0, 0]) for s in np.array(buffer.sample(100)[0])}
        self.assertEqual(markers, {2.0, 3.0})

    def test_sampling_covers_every_generation(self):
        buffer = ReplayBuffer(capacity=100, policy_output_dim=2, seed=1)
        for marker in range(1, 4):
            history = [(position(True, marker), mx.array([1.0, 0.0]), None)] * 5
            buffer.add([(history, 0.0)])
        markers = {float(s[0, 0, 0]) for s in np.array(buffer.sample(100)[0])}
        self.assertEqual(markers, {1.0, 2.0, 3.0})

    def test_empty_buffer_cannot_be_sampled(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(capacity=1, policy_output_dim=2).sample(1)


if __name__ == "__main__":
    unittest.main()
