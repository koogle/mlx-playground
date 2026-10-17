"""Bounded store of recent self-play positions for repeated minibatch training."""

from collections import deque
from typing import Iterable, Tuple

import mlx.core as mx
import numpy as np


def side_to_move_value(state, white_result: float) -> float:
    """Convert a White-perspective result to the perspective of the side to move.

    Plane 12 is 1 for White to move and 0 for Black to move.
    """
    if float(state[12, 0, 0]) == 1.0:
        return float(white_result)
    return -float(white_result)


class ReplayBuffer:
    """Keep the most recent positions, evicting whole generations when full.

    Policies are stored sparsely (legal move indices and probabilities) and
    densified per minibatch, so a 4672-way target costs a few dozen floats.
    """

    def __init__(self, capacity: int, policy_output_dim: int, seed=None):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.policy_output_dim = policy_output_dim
        self._generations = deque()
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    @property
    def n_generations(self) -> int:
        return len(self._generations)

    def add(self, games: Iterable[Tuple[list, float]]) -> int:
        """Store every position of a generation; returns how many were added."""
        states, indices, probabilities, values = [], [], [], []
        for history, white_result in games:
            for state, policy, _ in history:
                state_np = np.asarray(state, dtype=np.float32)
                policy_np = np.asarray(policy, dtype=np.float32)
                nonzero = np.flatnonzero(policy_np)
                states.append(state_np.astype(np.float16))
                indices.append(nonzero.astype(np.int32))
                probabilities.append(policy_np[nonzero])
                values.append(side_to_move_value(state_np, white_result))
        if not states:
            return 0
        self._generations.append(
            (np.stack(states), indices, probabilities, np.array(values, np.float32))
        )
        self._size += len(states)
        while self._size > self.capacity and len(self._generations) > 1:
            self._size -= len(self._generations.popleft()[0])
        return len(states)

    def sample(self, batch_size: int):
        """Draw one uniformly sampled minibatch as MLX arrays."""
        if not self._size:
            raise ValueError("Cannot sample from an empty replay buffer")
        batch_size = min(batch_size, self._size)
        flat = self._rng.integers(0, self._size, size=batch_size)
        flat.sort()
        states = np.empty((batch_size, *self._generations[0][0].shape[1:]), np.float32)
        policies = np.zeros((batch_size, self.policy_output_dim), np.float32)
        values = np.empty(batch_size, np.float32)
        offset = 0
        position = 0
        for gen_states, gen_indices, gen_probabilities, gen_values in self._generations:
            end = offset + len(gen_states)
            while position < batch_size and flat[position] < end:
                local = flat[position] - offset
                states[position] = gen_states[local]
                policies[position, gen_indices[local]] = gen_probabilities[local]
                values[position] = gen_values[local]
                position += 1
            offset = end
        return mx.array(states), mx.array(policies), mx.array(values)

    def sample_batches(self, batch_size: int, n_batches: int):
        for _ in range(n_batches):
            yield self.sample(batch_size)
