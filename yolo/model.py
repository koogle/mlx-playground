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


class DarkNet19(nn.Module):
    """Darknet-19 backbone with configurable output size"""

    def __init__(self, target_size: int = 7):
        super().__init__()
        self.target_size = target_size
        self.relu = nn.ReLU()

        # Initial convolution layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm(128)
        self.conv3_2 = nn.Conv2d(128, 64, kernel_size=1)
        self.bn3_2 = nn.BatchNorm(64)
        self.conv3_3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 4
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm(256)
        self.conv4_2 = nn.Conv2d(256, 128, kernel_size=1)
        self.bn4_2 = nn.BatchNorm(128)
        self.conv4_3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 5
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm(512)
        self.conv5_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn5_2 = nn.BatchNorm(256)
        self.conv5_3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5_3 = nn.BatchNorm(512)
        self.conv5_4 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn5_4 = nn.BatchNorm(256)
        self.conv5_5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5_5 = nn.BatchNorm(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 6
        self.conv6_1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn6_1 = nn.BatchNorm(1024)
        self.conv6_2 = nn.Conv2d(1024, 512, kernel_size=1)
        self.bn6_2 = nn.BatchNorm(512)
        self.conv6_3 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn6_3 = nn.BatchNorm(1024)
        self.conv6_4 = nn.Conv2d(1024, 512, kernel_size=1)
        self.bn6_4 = nn.BatchNorm(512)
        self.conv6_5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn6_5 = nn.BatchNorm(1024)

        # Calculate final feature map size and add pooling if needed
        curr_size = 14  # Size after all conv layers
        if curr_size > target_size:
            pool_size = curr_size // target_size
            if pool_size > 1:
                self.final_pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
            else:
                self.final_pool = None
        else:
            self.final_pool = None

    def __call__(self, x):
        # Initial conv
        x = self.relu(self.bn1(self.conv1(x)))  # 448 -> 448
        x = self.pool1(x)  # 448 -> 224

        # Layer 2
        x = self.relu(self.bn2(self.conv2(x)))  # 224 -> 224
        x = self.pool2(x)  # 224 -> 112

        # Layer 3
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.relu(self.bn3_3(self.conv3_3(x)))
        x = self.pool3(x)  # 112 -> 56

        # Layer 4
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        x = self.relu(self.bn4_3(self.conv4_3(x)))
        x = self.pool4(x)  # 56 -> 28

        # Layer 5
        x = self.relu(self.bn5_1(self.conv5_1(x)))
        x = self.relu(self.bn5_2(self.conv5_2(x)))
        x = self.relu(self.bn5_3(self.conv5_3(x)))
        x = self.relu(self.bn5_4(self.conv5_4(x)))
        x = self.relu(self.bn5_5(self.conv5_5(x)))
        x = self.pool5(x)  # 28 -> 14

        # Layer 6
        x = self.relu(self.bn6_1(self.conv6_1(x)))
        x = self.relu(self.bn6_2(self.conv6_2(x)))
        x = self.relu(self.bn6_3(self.conv6_3(x)))
        x = self.relu(self.bn6_4(self.conv6_4(x)))
        x = self.relu(self.bn6_5(self.conv6_5(x)))  # 14 -> 14

        # Apply final pooling if needed
        if self.final_pool is not None:
            x = self.final_pool(x)

        return x


class YOLO(nn.Module):
    """YOLOv2 object detection model"""

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

        # Create backbone with target size S
        self.backbone = DarkNet19(target_size=S)

        # Detection head
        self.detect1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn_detect1 = nn.BatchNorm(1024)
        self.detect2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn_detect2 = nn.BatchNorm(1024)

        # Final detection layer
        self.conv_final = nn.Conv2d(1024, B * (5 + C), kernel_size=1)

        # Activation
        self.relu = nn.ReLU()

        # Initialize anchor boxes
        self.anchors = mx.array([
            [1.3221, 1.73145],
            [3.19275, 4.00944],
        ])

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
        x = self.backbone(x)  # SxSx1024

        # Detection head
        x = self.relu(self.bn_detect1(self.detect1(x)))
        x = self.relu(self.bn_detect2(self.detect2(x)))
        x = self.conv_final(x)

        # Reshape to (batch_size, S, S, B * (5 + C))
        x = mx.transpose(x, (0, 2, 3, 1))
        x = mx.reshape(x, (batch_size, self.S, self.S, self.B * (5 + self.C)))

        return x

    def decode_predictions(self, pred):
        """Decode raw predictions to bounding boxes"""
        # Implementation remains the same
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
