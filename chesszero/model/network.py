import mlx.core as mx
import mlx.nn as nn
from config.model_config import ModelConfig


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
            config.input_shape[-1], config.n_filters, kernel_size=3, padding=1
        )
        self.bn_input = nn.BatchNorm(config.n_filters)

        # Residual tower
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(config.n_filters) for _ in range(config.n_residual_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(config.n_filters, 32, kernel_size=3, padding=1)
        self.policy_bn = nn.BatchNorm(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, config.policy_output_dim)

        # Value head
        self.value_conv = nn.Conv2d(config.n_filters, 32, kernel_size=3, padding=1)
        self.value_bn = nn.BatchNorm(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def __call__(self, x):
        # Input block
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = nn.relu(x)

        # Residual tower
        x = self.residual_tower(x)

        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = nn.relu(policy)
        policy = policy.reshape(-1, 32 * 8 * 8)
        policy = self.policy_fc(policy)
        policy = nn.softmax(policy)

        # Value head
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = nn.relu(value)
        value = value.reshape(-1, 32 * 8 * 8)
        value = self.value_fc1(value)
        value = nn.relu(value)
        value = self.value_fc2(value)
        value = nn.tanh(value)

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
