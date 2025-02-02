import mlx.core as mx
import mlx.nn as nn
from config.model_config import ModelConfig
from typing import Tuple
from mlx.utils import tree_map_with_path
import json
from pathlib import Path


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and skip connection"""

    def __init__(self, n_filters: int):
        super().__init__()
        # First convolution
        self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(n_filters)

        # Second convolution
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(n_filters)

        self.relu = nn.ReLU()
        self.debug = False

        # Clip values to prevent explosion
        self.clip_value = 10.0

    def __call__(self, x: mx.array) -> mx.array:
        identity = x

        if self.debug:
            print("\nResBlock input:", x.shape)
            print("Input has NaN:", mx.isnan(x).any())

        # First conv block
        out = self.conv1(x)
        out = mx.clip(out, -self.clip_value, self.clip_value)
        if self.debug:
            print(
                "After conv1:",
                mx.isnan(out).any(),
                "min:",
                out.min(),
                "max:",
                out.max(),
            )

        out = self.bn1(out)
        out = mx.clip(out, -self.clip_value, self.clip_value)
        if self.debug:
            print(
                "After bn1:", mx.isnan(out).any(), "min:", out.min(), "max:", out.max()
            )

        out = self.relu(out)
        if self.debug:
            print("After relu1:", mx.isnan(out).any())

        # Second conv block
        out = self.conv2(out)
        out = mx.clip(out, -self.clip_value, self.clip_value)
        if self.debug:
            print(
                "After conv2:",
                mx.isnan(out).any(),
                "min:",
                out.min(),
                "max:",
                out.max(),
            )

        out = self.bn2(out)
        out = mx.clip(out, -self.clip_value, self.clip_value)
        if self.debug:
            print(
                "After bn2:", mx.isnan(out).any(), "min:", out.min(), "max:", out.max()
            )

        # Skip connection with clipping
        out = out + identity
        out = mx.clip(out, -self.clip_value, self.clip_value)
        if self.debug:
            print(
                "After skip:", mx.isnan(out).any(), "min:", out.min(), "max:", out.max()
            )

        out = self.relu(out)
        if self.debug:
            print("After final relu:", mx.isnan(out).any())

        return out


class ChessNet(nn.Module):
    """Neural network combining policy and value networks"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Input convolution block
        self.conv_input = nn.Conv2d(
            in_channels=19,  # Updated for new BitBoard state (19 channels)
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
        """Initialize weights with Kaiming initialization"""

        def init_fn(path, a):
            if "conv" in path and "weight" in path:
                # Use Kaiming initialization for convolutions
                fan_in = (
                    a.shape[1] * a.shape[2] * a.shape[-1]
                )  # kernel_size * kernel_size * in_channels
                std = mx.sqrt(2.0 / fan_in)  # He initialization
                return mx.random.normal(a.shape) * std
            elif "linear" in path and "weight" in path:
                # Use Kaiming initialization for linear layers
                fan_in = a.shape[1]
                std = mx.sqrt(2.0 / fan_in)
                return mx.random.normal(a.shape) * std
            elif "bn" in path:
                if "weight" in path:  # gamma
                    return mx.ones(a.shape) * 0.1  # Start with smaller scale
                else:  # bias (beta)
                    return mx.zeros(a.shape)
            elif "bias" in path:
                return mx.zeros(a.shape)
            return a

        # Update all parameters using the initialization function
        self.update(tree_map_with_path(init_fn, self.parameters()))

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Forward pass with explicit evaluation of intermediate tensors"""
        # Input comes as [batch_size, channels, height, width]
        x = mx.transpose(x, (0, 2, 3, 1))  # to [batch_size, height, width, channels]

        # Input block
        x = self.conv_input(x)  # Now expects 19 input channels
        x = self.bn_input(x)
        x = self.relu(x)
        mx.eval(x)  # Evaluate after input block

        # Residual tower
        x = self.residual_tower(x)
        mx.eval(x)  # Evaluate after residual tower

        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.relu(policy)
        policy = mx.reshape(policy, (-1, 32 * 8 * 8))
        policy = self.policy_fc(policy)
        policy = mx.softmax(policy, axis=-1)
        mx.eval(policy)  # Evaluate policy output

        # Value head
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.relu(value)
        value = mx.reshape(value, (-1, 32 * 8 * 8))
        value = self.value_fc1(value)
        value = self.relu(value)
        value = self.value_fc2(value)
        value = mx.tanh(value)
        mx.eval(value)  # Evaluate value output

        # Clean up intermediate tensor x
        del x

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

    def save_checkpoint(self, checkpoint_dir: str, epoch: int, optimizer_state=None):
        """Save model checkpoint and training state"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model parameters
        model_path = checkpoint_dir / f"model_epoch_{epoch}.safetensors"

        # Save model weights
        self.save_weights(str(model_path))

        # Convert optimizer state arrays to regular Python lists for JSON serialization
        if optimizer_state is not None:
            optimizer_state = {
                "step": int(optimizer_state["step"]),  # Convert uint64 to int
                "learning_rate": float(
                    optimizer_state["learning_rate"]
                ),  # Convert to float
            }

        # Save training state
        state = {
            "epoch": epoch,
            "optimizer_state": optimizer_state,
            "config": vars(self.config),
        }
        state_path = checkpoint_dir / f"state_epoch_{epoch}.json"
        with open(state_path, "w") as f:
            json.dump(state, f)

    @classmethod
    def load_checkpoint(
        cls, checkpoint_dir: str, epoch: int
    ) -> Tuple["ChessNet", dict]:
        """Load model checkpoint and return model and training state"""
        checkpoint_dir = Path(checkpoint_dir)

        # Load model parameters
        model_path = checkpoint_dir / f"model_epoch_{epoch}.safetensors"

        # Load training state
        state_path = checkpoint_dir / f"state_epoch_{epoch}.json"
        with open(state_path) as f:
            state = json.load(f)

        # Convert optimizer state back to MLX arrays if it exists
        if state["optimizer_state"] is not None:
            state["optimizer_state"] = {
                "step": mx.array(state["optimizer_state"]["step"], dtype=mx.uint64),
                "learning_rate": mx.array(
                    state["optimizer_state"]["learning_rate"], dtype=mx.float32
                ),
            }

        # Recreate model with saved config
        config = ModelConfig(**state["config"])
        model = cls(config)
        model.load_weights(str(model_path))

        return model, state
