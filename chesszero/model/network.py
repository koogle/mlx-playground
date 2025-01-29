import mlx.core as mx
import mlx.nn as nn
from config.model_config import ModelConfig
from typing import Tuple
from mlx.utils import tree_map_with_path


class ResidualBlock(nn.Module):
    """Residual block for the chess network"""

    def __init__(self, n_filters: int):
        super().__init__()
        # Note: in_channels should match out_channels for residual connection
        self.conv1 = nn.Conv2d(
            in_channels=n_filters, out_channels=n_filters, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm(n_filters)
        self.conv2 = nn.Conv2d(
            in_channels=n_filters, out_channels=n_filters, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm(n_filters)

    def __call__(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = nn.relu(out)

        return out


class ChessNet(nn.Module):
    """Neural network combining policy and value networks"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Input convolution block
        self.conv_input = nn.Conv2d(
            in_channels=14,  # Input channels for chess board encoding
            out_channels=config.n_filters,
            kernel_size=3,
            padding=1,
        )
        self.bn_input = nn.BatchNorm(config.n_filters)
        self.relu = nn.ReLU()

        # Residual tower
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(config.n_filters) for _ in range(config.n_residual_blocks)]
        )

        # Policy head - reduce channels to 32
        self.policy_conv = nn.Conv2d(
            in_channels=config.n_filters, out_channels=32, kernel_size=1
        )
        self.policy_bn = nn.BatchNorm(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, config.policy_output_dim)

        # Value head - reduce channels to 32
        self.value_conv = nn.Conv2d(
            in_channels=config.n_filters, out_channels=32, kernel_size=1
        )
        self.value_bn = nn.BatchNorm(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Initialize weights
        self._init_weights()

        self.train()  # Set to training mode

    def _init_weights(self):
        """Initialize weights with small random values"""

        def init_fn(path, a):
            if "conv" in path and "weight" in path:
                # MLX Conv2d weights shape is (out_channels, kernel_h, kernel_w, in_channels)
                # For fan-in we want kernel_h * kernel_w * in_channels
                n = a.shape[1] * a.shape[2] * a.shape[-1]  # Use -1 to get in_channels
                return mx.random.normal(a.shape) / mx.sqrt(n)
            elif "linear" in path and "weight" in path:
                # Linear weights have shape (out_features, in_features)
                n = a.shape[1]  # fan-in
                return mx.random.normal(a.shape) / mx.sqrt(n)
            elif "bias" in path:
                return mx.zeros(a.shape)
            elif "bn" in path:  # BatchNorm parameters
                if "weight" in path:
                    return mx.ones(a.shape)
                else:  # bias
                    return mx.zeros(a.shape)
            return a  # Leave other parameters unchanged

        # Update all parameters using the initialization function
        self.update(tree_map_with_path(init_fn, self.parameters()))

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        # Input comes as [batch_size, channels, height, width] from encode_board
        x = mx.transpose(
            x, (0, 2, 3, 1)
        )  # [batch, channels, height, width] -> [batch, height, width, channels]

        # Input block
        x = self.conv_input(x)
        x = self.bn_input(x) if self.training else x  # Only use BatchNorm in training
        x = self.relu(x)

        # Residual tower
        x = self.residual_tower(x)

        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy) if self.training else policy
        policy = self.relu(policy)
        policy = mx.reshape(policy, (-1, 32 * 8 * 8))
        policy = self.policy_fc(policy)
        policy = mx.softmax(policy, axis=-1)

        # Value head
        value = self.value_conv(x)
        value = self.value_bn(value) if self.training else value
        value = self.relu(value)
        value = mx.reshape(value, (-1, 32 * 8 * 8))
        value = self.value_fc1(value)
        value = self.relu(value)
        value = self.value_fc2(value)
        value = mx.tanh(value)

        return policy, value

    def loss_fn(self, states, target_policies, target_values):
        """
        Calculate the combined policy and value loss

        Args:
            states: Batch of board states [batch_size, 14, 8, 8]
            target_policies: Target move probabilities [batch_size, policy_output_dim]
            target_values: Target game outcomes [batch_size, 1]

        Returns:
            Combined loss value
        """
        policies, values = self(states)

        # Policy loss (cross entropy)
        policy_loss = (
            -mx.sum(target_policies * mx.log(policies + 1e-8)) / policies.shape[0]
        )

        # Value loss (mean squared error)
        value_loss = mx.mean((target_values - values.squeeze()) ** 2)

        # Combined loss
        total_loss = policy_loss + value_loss

        return total_loss
