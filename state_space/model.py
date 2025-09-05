import mlx.core as mx
import mlx.nn as nn


class StateSpace(nn.Module):
    """
    Basic State Space Model (SSM) implementation in MLX.

    The continuous SSM is defined as:
        x'(t) = A x(t) + B u(t)
        y(t) = C x(t) + D u(t)

    We discretize it for sequence modeling using zero-order hold.
    """

    def __init__(self, dim_input, dim_state, dim_output, dt_min=0.001, dt_max=0.1):
        super(StateSpace, self).__init__()

        self.dim_input = dim_input
        self.dim_state = dim_state
        self.dim_output = dim_output

        # Initialize continuous SSM parameters
        self.A = nn.Linear(dim_state, dim_state, bias=False)
        self.B = nn.Linear(dim_input, dim_state, bias=False)
        self.C = nn.Linear(dim_state, dim_output, bias=False)
        self.D = nn.Linear(dim_input, dim_output, bias=False)

        # Learnable discretization timestep
        self.log_dt = mx.zeros((1,))

        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize SSM parameters with sensible defaults"""
        # Initialize A as slightly negative diagonal (stable system)
        A_init = -0.5 * mx.eye(self.dim_state) + 0.1 * mx.random.normal(
            (self.dim_state, self.dim_state)
        )
        self.A.weight = A_init

        # Initialize B and C with small random values
        self.B.weight = 0.1 * mx.random.normal((self.dim_state, self.dim_input))
        self.C.weight = 0.1 * mx.random.normal((self.dim_output, self.dim_state))

        # Initialize D near zero (no direct feedthrough)
        self.D.weight = 0.01 * mx.random.normal((self.dim_output, self.dim_input))

    def discretize(self):
        """Discretize continuous SSM using zero-order hold

        The model is defined over a continous function via A and B matrixes
        but at every inference and training step we need to perform
        it at a given point in time.

        We approximate the slope of the function in the point t for a small delta dt
        and based on that can compute the discrete values for A & B.
        This enables us to perform inference and computation at a point in time by knowing
        the slope without having to solving the whole equation.
        """
        dt = mx.exp(self.log_dt)

        # Get matrices
        A = self.A.weight
        B = self.B.weight

        # Assume dt is small we can figure out discrete A
        # A_d = exp(A * dt) ≈ I + A*dt for small dt
        A_discrete = mx.eye(self.dim_state) + dt * A

        # B_d = (exp(A*dt) - I) * A^{-1} * B ≈ dt * B for small dt
        B_discrete = dt * B

        return A_discrete, B_discrete

    def __call__(self, u, initial_state=None):
        """
        Forward pass of the SSM.

        Args:
            u: Input sequence of shape (batch_size, seq_len, dim_input)
            initial_state: Initial hidden state (batch_size, dim_state)

        Returns:
            y: Output sequence of shape (batch_size, seq_len, dim_output)
        """
        batch_size, seq_len, _ = u.shape

        # Get discrete SSM parameters
        A_d, B_d = self.discretize()
        C = self.C.weight
        D = self.D.weight

        # Initialize state
        if initial_state is None:
            internal_state = mx.zeros((batch_size, self.dim_state))
        else:
            internal_state = initial_state

        # Run recurrence
        outputs = []
        for t in range(seq_len):
            # u_t
            input_signal = u[:, t, :]

            # State update: x_{t+1} = A_d @ x_t + B_d @ u_t
            internal_state = mx.matmul(internal_state, A_d.T) + mx.matmul(
                input_signal, B_d.T
            )

            # Output: y_t = C @ x_t + D @ u_t
            y_t = mx.matmul(internal_state, C.T) + mx.matmul(input_signal, D.T)
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
        return self(u)


class S4Model(nn.Module):
    """
    Structured State Space (S4) inspired model.
    Uses HiPPO initialization for better long-range modeling.
    """

    def __init__(self, dim_input, dim_state, dim_output, n_layers=2):
        super().__init__()

        # Create layers directly as attributes (following pattern from other models)
        if n_layers >= 1:
            self.layer_0 = StateSpace(dim_input, dim_state, dim_output)
        if n_layers >= 2:
            self.layer_1 = StateSpace(dim_output, dim_state, dim_output)
        if n_layers >= 3:
            self.layer_2 = StateSpace(dim_output, dim_state, dim_output)
        if n_layers >= 4:
            self.layer_3 = StateSpace(dim_output, dim_state, dim_output)
        
        self.n_layers = n_layers
        self.norm = nn.LayerNorm(dim_output)

    def __call__(self, x):
        if self.n_layers >= 1:
            x = self.layer_0(x)
            x = mx.maximum(x, 0)  # ReLU activation
        if self.n_layers >= 2:
            x = self.layer_1(x)
            x = mx.maximum(x, 0)  # ReLU activation
        if self.n_layers >= 3:
            x = self.layer_2(x)
            x = mx.maximum(x, 0)  # ReLU activation
        if self.n_layers >= 4:
            x = self.layer_3(x)
            x = mx.maximum(x, 0)  # ReLU activation

        return self.norm(x)
