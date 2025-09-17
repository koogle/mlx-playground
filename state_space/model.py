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

    def __init__(self, dim_input, dim_state, dim_output, dt_min=0.001, dt_max=0.1, init_strategy="standard"):
        super(StateSpace, self).__init__()

        self.dim_input = dim_input
        self.dim_state = dim_state
        self.dim_output = dim_output
        self.init_strategy = init_strategy  # "standard", "improved", "hippo"

        # Initialize continuous SSM parameters
        self.A = nn.Linear(dim_state, dim_state, bias=False)
        self.B = nn.Linear(dim_input, dim_state, bias=False)
        self.C = nn.Linear(dim_state, dim_output, bias=False)
        self.D = nn.Linear(dim_input, dim_output, bias=False)

        # Fixed small discretization timestep (not learnable for stability)
        # Use smaller dt for improved strategy
        self.dt = 0.005 if self.init_strategy == "improved" else 0.01

        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize SSM parameters with selected strategy"""
        if self.init_strategy == "hippo":
            # HiPPO initialization for better long-range modeling
            A_init = self._hippo_initialization()
        elif self.init_strategy == "improved":
            # Improved initialization - stronger negative diagonal for stability
            # and smaller noise to reduce variance
            A_init = -1.0 * mx.eye(self.dim_state) + 0.01 * mx.random.normal(
                (self.dim_state, self.dim_state)
            )
        else:  # "standard"
            # Standard initialization - slightly negative diagonal (stable system)
            A_init = -0.5 * mx.eye(self.dim_state) + 0.1 * mx.random.normal(
                (self.dim_state, self.dim_state)
            )

        self.A.weight = A_init

        # Initialize B and C based on strategy
        if self.init_strategy == "improved":
            # Xavier/Glorot initialization for better gradient flow
            fan_in_b = self.dim_input
            fan_out_b = self.dim_state
            std_b = mx.sqrt(2.0 / (fan_in_b + fan_out_b))
            self.B.weight = std_b * mx.random.normal((self.dim_state, self.dim_input))

            fan_in_c = self.dim_state
            fan_out_c = self.dim_output
            std_c = mx.sqrt(2.0 / (fan_in_c + fan_out_c))
            self.C.weight = std_c * mx.random.normal((self.dim_output, self.dim_state))

            # Keep D very small for minimal direct feedthrough
            self.D.weight = 0.001 * mx.random.normal((self.dim_output, self.dim_input))
        else:
            # Standard/HiPPO initialization for B, C, D
            self.B.weight = 0.1 * mx.random.normal((self.dim_state, self.dim_input))
            self.C.weight = 0.1 * mx.random.normal((self.dim_output, self.dim_state))
            self.D.weight = 0.01 * mx.random.normal((self.dim_output, self.dim_input))

    def _hippo_initialization(self):
        """Stable initialization for the A matrix"""
        n = self.dim_state
        # Create a stable, learnable matrix
        # Negative diagonal for stability + small random off-diagonals
        A = -1.0 * mx.eye(n)  # Stable diagonal
        # Add small random perturbations to break symmetry
        A = A + 0.01 * mx.random.normal((n, n))
        return A

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
        dt = self.dt

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
            # Shapes: internal_state: (batch, dim_state), A_d: (dim_state, dim_state)
            # Need: (batch, dim_state) = (dim_state, dim_state) @ (dim_state, batch).T -> (batch, dim_state)
            internal_state = (A_d @ internal_state.T).T + (B_d @ input_signal.T).T

            # Output: y_t = C @ x_t + D @ u_t  
            y_t = (C @ internal_state.T).T + (D @ input_signal.T).T
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

    def __init__(self, dim_input, dim_state, dim_output, n_layers=2, init_strategy="standard"):
        super().__init__()

        layers = []

        # First layer
        if n_layers > 0:
            layers.append(StateSpace(dim_input, dim_state, dim_output, init_strategy=init_strategy))
            layers.append(nn.ReLU())

        # Additional layers
        for _ in range(n_layers - 1):
            layers.append(StateSpace(dim_output, dim_state, dim_output, init_strategy=init_strategy))
            layers.append(nn.ReLU())
        
        self.layers = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(dim_output)

    def __call__(self, x):
        x = self.layers(x)
        return self.norm(x)
