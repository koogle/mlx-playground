import mlx.core as mx
import mlx.nn as nn
from config.model_config import ModelConfig
from typing import Tuple


class ResidualBlock(nn.Module):
    """Residual block for the chess network"""

    def __init__(self, n_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
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
            14,  # Fixed number of input channels for chess board encoding
            config.n_filters,
            kernel_size=3,
            padding=1,
        )
        self.bn_input = nn.BatchNorm(config.n_filters)
        self.relu = nn.ReLU()

        # Residual tower
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(config.n_filters) for _ in range(config.n_residual_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(config.n_filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, config.policy_output_dim)

        # Value head
        self.value_conv = nn.Conv2d(config.n_filters, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        # Ensure input is the right shape [batch_size, channels, height, width]
        if len(x.shape) == 3:
            x = mx.expand_dims(x, axis=0)  # Add batch dimension

        # Convert to NCHW format if in NHWC
        if x.shape[-1] == 14:  # If channels are last
            x = mx.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW

        # Input block
        x = self.relu(self.bn_input(self.conv_input(x)))

        # Residual tower
        x = self.residual_tower(x)

        # Policy head
        policy = self.relu(self.policy_bn(self.policy_conv(x)))
        policy = mx.reshape(policy, (-1, 32 * 8 * 8))
        policy = self.policy_fc(policy)
        policy = mx.softmax(policy, axis=-1)

        # Value head
        value = self.relu(self.value_bn(self.value_conv(x)))
        value = mx.reshape(value, (-1, 32 * 8 * 8))
        value = self.relu(self.value_fc1(value))
        value = mx.tanh(self.value_fc2(value))

        return policy, value

    def loss_fn(self, states, target_policies, target_values):
        """
        Calculate the combined policy and value loss

        Args:
            states: Batch of board states
            target_policies: Target move probabilities
            target_values: Target game outcomes

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
