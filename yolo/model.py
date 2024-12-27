import mlx.core as mx
import mlx.nn as nn

import mlx.nn.layers.pooling as pool


class DarkNet(nn.Module):
    """
    The backbone network for YOLO, based on GoogLeNet architecture
    but using simpler 1x1 reduction layers followed by 3x3 conv layers
    """

    def __init__(self):
        super().__init__()
        # Initial convolution layers
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self.bn1 = nn.BatchNorm(64)

        self.pool1 = pool.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Conv layers with 1x1 reduction
        self.conv2_1 = nn.Conv2d(
            in_channels=64, out_channels=192, kernel_size=3, padding=1
        )
        self.bn2_1 = nn.BatchNorm(192)
        self.pool2 = pool.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Deeper layers
        self.conv3_1 = nn.Conv2d(192, 128, kernel_size=1)
        self.conv3_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv3_4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm(512)
        self.pool3 = pool.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Fourth block (similar pattern with more filters)
        self.conv4_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv4_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv4_4 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm(1024)
        self.pool4 = pool.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Fifth block
        self.conv5_1 = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv5_2 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm(1024)

        self.relu = nn.ReLU()

    def __call__(self, x):
        # Input should be in NHWC format (batch, height, width, channels)
        # MLX Conv2d weights are in (out_channels, kernel_h, kernel_w, in_channels) format

        # Initial convolutions
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Second block
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.pool2(x)

        # Third block with 1x1 reductions
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.relu(self.bn3(self.conv3_4(x)))
        x = self.pool3(x)

        # Fourth block
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.relu(self.bn4(self.conv4_4(x)))
        x = self.pool4(x)

        # Fifth block
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.relu(self.bn5(self.conv5_4(x)))

        return x


class YOLO(nn.Module):
    """
    YOLO (You Only Look Once) object detection model
    S x S grid cells, each predicting B bounding boxes with confidence
    and C class probabilities
    """

    def __init__(self, S=7, B=2, C=20):
        """
        Args:
            S: Grid size (S x S)
            B: Number of bounding boxes per grid cell
            C: Number of classes
        """
        super().__init__()
        self.S = S
        self.B = B
        self.C = C

        self.backbone = DarkNet()

        # Detection layers
        self.conv6 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm(1024)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm(1024)

        # Fully connected layers
        self.flatten = lambda x: mx.reshape(x, (x.shape[0], -1))
        self.fc1 = nn.Linear(7 * 7 * 1024, 4096)

        self.dropout = nn.Dropout(p=0.5) if self.training else lambda x: x
        self.fc2 = nn.Linear(4096, S * S * (B * 5 + C))
        self.relu = nn.ReLU()

    def __call__(self, x, return_features=False):
        # Get features from backbone
        x = self.backbone(x)

        # Detection head
        conv6_features = self.relu(self.bn6(self.conv6(x)))
        conv7_features = self.relu(self.bn7(self.conv7(conv6_features)))

        # Flatten and FC layers
        x = self.flatten(conv7_features)
        fc1_features = self.relu(self.fc1(x))
        x = self.dropout(fc1_features)
        x = self.fc2(x)

        # Reshape to match grid format [batch, S, S, B*5 + C]
        batch_size = x.shape[0]
        x = mx.reshape(x, (batch_size, self.S, self.S, self.B * 5 + self.C))

        if return_features:
            return x, {
                'conv6': conv6_features,
                'conv7': conv7_features,
                'fc1': fc1_features
            }
        return x
