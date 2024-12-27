"""
YOLOv2 Implementation with configurable backbone architecture.
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ConvConfig:
    """Configuration for a convolutional layer"""

    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: Optional[int] = None

    def __post_init__(self):
        if self.padding is None:
            self.padding = self.kernel_size // 2


class DarkNetBlock(nn.Module):
    """A configurable DarkNet block with conv, batch norm, and ReLU"""

    def __init__(self, in_channels: int, config: ConvConfig):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            config.out_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
        )
        self.bn = nn.BatchNorm(config.out_channels)
        self.relu = nn.ReLU()

    def __call__(self, x):
        return self.relu(self.bn(self.conv(x)))


class DarkNetBackbone(nn.Module):
    """
    Configurable DarkNet backbone

    Args:
        config: List[ConvConfig]: Layer configurations
        input_channels: int: Number of input channels (default: 3 for RGB)
        target_size: int: Target output size (e.g., 7 for 7x7 feature maps)
        input_size: int: Input image size (default: 448 for YOLOv2)
    """

    def __init__(
        self,
        config: List[ConvConfig],
        input_channels: int = 3,
        target_size: int = 7,
        input_size: int = 448,
    ):
        super().__init__()
        self.target_size = target_size
        self.input_size = input_size
        self.relu = nn.ReLU()

        # Build backbone layers directly as attributes
        in_channels = input_channels
        curr_size = input_size
        
        for i, conv_config in enumerate(config):
            # Create conv and bn layers directly as attributes
            setattr(self, f"conv_{i}", nn.Conv2d(
                in_channels,
                conv_config.out_channels,
                kernel_size=conv_config.kernel_size,
                stride=conv_config.stride,
                padding=conv_config.padding,
            ))
            setattr(self, f"bn_{i}", nn.BatchNorm(conv_config.out_channels))
            
            in_channels = conv_config.out_channels
            curr_size = curr_size // conv_config.stride

        self.num_layers = len(config)

        # Add final pooling if needed
        if curr_size > target_size:
            pool_size = curr_size // target_size
            if pool_size > 1:
                self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
            else:
                self.pool = None
        else:
            self.pool = None

    def __call__(self, x):
        # Direct layer-to-layer connections
        for i in range(self.num_layers):
            conv = getattr(self, f"conv_{i}")
            bn = getattr(self, f"bn_{i}")
            x = self.relu(bn(conv(x)))

        if self.pool is not None:
            x = self.pool(x)

        return x


class YOLO(nn.Module):
    """
    YOLOv2 model with configurable backbone

    The model outputs predictions in the format:
    [tx, ty, tw, th, to] for each bounding box
    and C class probabilities
    """

    def __init__(self, S=7, B=2, C=20):
        """
        Args:
            S: Grid size (S x S)
            B: Number of bounding boxes per cell
            C: Number of classes
        """
        super().__init__()
        self.S = S
        self.B = B
        self.C = C

        # Define backbone configuration
        backbone_config = [
            # Initial layers (448 -> 224)
            ConvConfig(32),
            ConvConfig(64, stride=2),
            # Downsample to 112
            ConvConfig(128),
            ConvConfig(64, kernel_size=1),
            ConvConfig(128, stride=2),
            # Downsample to 56
            ConvConfig(256),
            ConvConfig(128, kernel_size=1),
            ConvConfig(256, stride=2),
            # Downsample to 28
            ConvConfig(512),
            ConvConfig(256, kernel_size=1),
            ConvConfig(512),
            ConvConfig(256, kernel_size=1),
            ConvConfig(512, stride=2),
            # Final feature extraction
            ConvConfig(1024),
            ConvConfig(512, kernel_size=1),
            ConvConfig(1024),
            ConvConfig(512, kernel_size=1),
            ConvConfig(1024),
        ]

        # Create backbone
        self.backbone = DarkNetBackbone(
            config=backbone_config, input_channels=3, target_size=S, input_size=448
        )

        # Detection head
        self.detect1 = DarkNetBlock(1024, ConvConfig(1024))
        self.detect2 = DarkNetBlock(1024, ConvConfig(1024))

        # Final detection layer
        self.conv_final = nn.Conv2d(1024, B * (5 + C), kernel_size=1)

        # Initialize anchor boxes
        self.anchors = mx.array(
            [
                [1.3221, 1.73145],
                [3.19275, 4.00944],
            ]
        )

    def __call__(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Output tensor of shape (batch_size, S, S, B * (5 + C))
            Each cell contains B bounding boxes with 5 coordinates (tx, ty, tw, th, confidence)
            and C class probabilities
        """
        batch_size = x.shape[0]

        # Backbone
        x = self.backbone(x)

        # Detection head
        x = self.detect1(x)
        x = self.detect2(x)
        x = self.conv_final(x)

        # Reshape to (batch_size, S, S, B * (5 + C))
        x = mx.transpose(x, (0, 2, 3, 1))
        x = mx.reshape(x, (batch_size, self.S, self.S, self.B * (5 + self.C)))

        return x

    def decode_predictions(self, pred):
        """
        Convert network output to bounding boxes
        Args:
            pred: Raw network output [batch, S, S, B * (5 + C)]
        Returns:
            boxes: [batch, S, S, B, 4] - normalized box coordinates (cx, cy, w, h)
            confidence: [batch, S, S, B] - objectness scores
            class_probs: [batch, S, S, B, C] - class probabilities
        """
        batch_size = pred.shape[0]

        # Reshape to [batch, S, S, B, 5 + C]
        pred = mx.reshape(pred, (batch_size, self.S, self.S, self.B, 5 + self.C))

        # Split prediction into components
        box_xy = mx.sigmoid(pred[..., 0:2])  # tx, ty -> sigmoid for [0,1]
        box_wh = mx.exp(pred[..., 2:4])  # tw, th -> exp for scaling
        conf = mx.sigmoid(pred[..., 4:5])  # to -> sigmoid for [0,1]
        prob = mx.softmax(pred[..., 5:], axis=-1)  # class probabilities

        # Add cell offsets to xy predictions
        grid_x = mx.arange(self.S, dtype=mx.float32)
        grid_y = mx.arange(self.S, dtype=mx.float32)
        grid_x, grid_y = mx.meshgrid(grid_x, grid_y)
        grid_x = mx.expand_dims(grid_x, axis=-1)
        grid_y = mx.expand_dims(grid_y, axis=-1)

        box_xy = (box_xy + mx.stack([grid_x, grid_y], axis=-1)) / self.S

        # Scale wh by anchors
        box_wh = box_wh * self.anchors

        # Combine xy and wh
        boxes = mx.concatenate([box_xy, box_wh], axis=-1)

        return boxes, conf, prob
