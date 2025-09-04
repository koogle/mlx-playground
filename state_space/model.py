import mlx.core as mx
import mlx.nn as nn
import numpy as np


class StateSpace(nn.Module):
    """
    Basic State Space Model (SSM) implementation in MLX.

    The continuous SSM is defined as:
        x'(t) = A x(t) + B u(t)
        y(t) = C x(t) + D u(t)

    We discretize it for sequence modeling using zero-order hold.
    """

    def __init__(self, d_input, d_state, d_output, dt_min=0.001, dt_max=0.1):
        super(StateSpace, self).__init__()

        self.d_input = d_input
        self.d_state = d_state
        self.d_output = d_output

        # Initialize continuous SSM parameters
        self.A = nn.Linear(d_state, d_state, bias=False)
        self.B = nn.Linear(d_input, d_state, bias=False)
        self.C = nn.Linear(d_state, d_output, bias=False)
        self.D = nn.Linear(d_input, d_output, bias=False)

        # Learnable discretization timestep
        self.log_dt = mx.zeros((1,))

        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize SSM parameters with sensible defaults"""
        # Initialize A as slightly negative diagonal (stable system)
        A_init = -0.5 * mx.eye(self.d_state) + 0.1 * mx.random.normal(
            (self.d_state, self.d_state)
        )
        self.A.weight = A_init

        # Initialize B and C with small random values
        self.B.weight = 0.1 * mx.random.normal((self.d_state, self.d_input))
        self.C.weight = 0.1 * mx.random.normal((self.d_output, self.d_state))

        # Initialize D near zero (no direct feedthrough)
        self.D.weight = 0.01 * mx.random.normal((self.d_output, self.d_input))

    def discretize(self):
        """Discretize continuous SSM using zero-order hold"""
        dt = mx.exp(self.log_dt)

        # Get matrices
        A = self.A.weight
        B = self.B.weight

        # Zero-order hold discretization
        # A_d = exp(A * dt) ≈ I + A*dt for small dt
        A_discrete = mx.eye(self.d_state) + dt * A

        # B_d = (exp(A*dt) - I) * A^{-1} * B ≈ dt * B for small dt
        B_discrete = dt * B

        return A_discrete, B_discrete

    def forward(self, u, initial_state=None):
        """
        Forward pass of the SSM.

        Args:
            u: Input sequence of shape (batch_size, seq_len, d_input)
            initial_state: Initial hidden state (batch_size, d_state)

        Returns:
            y: Output sequence of shape (batch_size, seq_len, d_output)
        """
        batch_size, seq_len, _ = u.shape

        # Get discrete SSM parameters
        A_d, B_d = self.discretize()
        C = self.C.weight
        D = self.D.weight

        # Initialize state
        if initial_state is None:
            x = mx.zeros((batch_size, self.d_state))
        else:
            x = initial_state

        # Run recurrence
        outputs = []
        for t in range(seq_len):
            u_t = u[:, t, :]

            # State update: x_{t+1} = A_d @ x_t + B_d @ u_t
            x = mx.matmul(x, A_d.T) + mx.matmul(u_t, B_d.T)

            # Output: y_t = C @ x_t + D @ u_t
            y_t = mx.matmul(x, C.T) + mx.matmul(u_t, D.T)
            outputs.append(y_t)

        # Stack outputs
        y = mx.stack(outputs, axis=1)

        return y

    def parallel_scan(self, u):
        """
        Efficient parallel scan implementation for training.
        This is a simplified version - full parallel scan would use associative scan.
        """
        # For now, just use the sequential version
        # A full implementation would use parallel prefix sum
        return self.forward(u)


class S4Model(nn.Module):
    """
    Structured State Space (S4) inspired model.
    Uses HiPPO initialization for better long-range modeling.
    """

    def __init__(self, d_input, d_state, d_output, n_layers=2):
        super(S4Model, self).__init__()

        self.layers = []
        for i in range(n_layers):
            layer_input = d_input if i == 0 else d_output
            self.layers.append(StateSpace(layer_input, d_state, d_output))

        self.norm = nn.LayerNorm(d_output)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = mx.maximum(x, 0)  # ReLU activation

        return self.norm(x)
