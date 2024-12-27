import mlx.core as mx
import mlx.nn as nn

import mlx.nn.layers.pooling as pool


class DarkNet(nn.Module):
    """The backbone network for YOLO, based on Darknet-19 architecture
    but using simpler 1x1 reduction layers followed by 3x3 conv layers
    """

    def __init__(self):
        super().__init__()
        # Initial convolution layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(32)
        self.pool1 = pool.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(64)
        self.pool2 = pool.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Layer 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm(128)
        self.conv3_2 = nn.Conv2d(128, 64, kernel_size=1)
        self.bn3_2 = nn.BatchNorm(64)
        self.conv3_3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm(128)
        self.pool3 = pool.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Layer 4
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm(256)
        self.conv4_2 = nn.Conv2d(256, 128, kernel_size=1)
        self.bn4_2 = nn.BatchNorm(128)
        self.conv4_3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm(256)
        self.pool4 = pool.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

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
        self.pool5 = pool.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

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

        self.relu = nn.ReLU()

    def __call__(self, x):
        # Initial conv
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Layer 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Layer 3
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.relu(self.bn3_3(self.conv3_3(x)))
        x = self.pool3(x)

        # Layer 4
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        x = self.relu(self.bn4_3(self.conv4_3(x)))
        route = x  # Save for passthrough layer
        x = self.pool4(x)

        # Layer 5
        x = self.relu(self.bn5_1(self.conv5_1(x)))
        x = self.relu(self.bn5_2(self.conv5_2(x)))
        x = self.relu(self.bn5_3(self.conv5_3(x)))
        x = self.relu(self.bn5_4(self.conv5_4(x)))
        x = self.relu(self.bn5_5(self.conv5_5(x)))
        x = self.pool5(x)

        # Layer 6
        x = self.relu(self.bn6_1(self.conv6_1(x)))
        x = self.relu(self.bn6_2(self.conv6_2(x)))
        x = self.relu(self.bn6_3(self.conv6_3(x)))
        x = self.relu(self.bn6_4(self.conv6_4(x)))
        x = self.relu(self.bn6_5(self.conv6_5(x)))

        return x, route


class YOLO(nn.Module):
    """
    YOLOv2 object detection model
    S x S grid cells, each predicting B bounding boxes with confidence
    and C class probabilities
    """

    def __init__(self, S=13, B=5, C=20):
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

        # Pre-computed anchor boxes from k-means clustering (normalized)
        # These values are from the YOLOv2 paper for VOC dataset
        self.anchors = mx.array([
            [1.08, 1.19],  # width, height
            [3.42, 4.41],
            [6.63, 11.38],
            [9.42, 5.11],
            [16.62, 10.52]
        ])

        self.backbone = DarkNet()

        # Passthrough layer - reorg the route tensor
        self.reorg = lambda x: mx.reshape(
            mx.transpose(
                mx.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 2, x.shape[3] // 2)),
                (0, 1, 2, 4, 3)
            ),
            (x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3] * 4)
        )

        # Detection layers
        self.conv6 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm(1024)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm(1024)

        # Combine fine-grained features
        self.conv_passthrough = nn.Conv2d(256, 64, kernel_size=1)
        self.bn_passthrough = nn.BatchNorm(64)

        # Final detection layer
        # For each cell: B * (5 + C) outputs
        # 5 = [tx, ty, tw, th, to] for each box
        self.conv_final = nn.Conv2d(1024 + 256, B * (5 + C), kernel_size=1)
        self.relu = nn.ReLU()

    def __call__(self, x, return_features=False):
        # Get features from backbone
        x, route = self.backbone(x)  # route is from conv4_3 (256 channels, 52x52)
        
        # Detection head
        conv6_features = self.relu(self.bn6(self.conv6(x)))  # 1024 channels, 13x13
        conv7_features = self.relu(self.bn7(self.conv7(conv6_features)))  # 1024 channels, 13x13

        # Process passthrough layer (space-to-depth)
        route = self.relu(self.bn_passthrough(self.conv_passthrough(route)))  # 64 channels
        route = self.reorg(route)  # 256 channels (64*4), halved spatial dimensions

        # Concatenate passthrough features with conv7
        x = mx.concatenate([route, conv7_features], axis=3)  # 1280 channels (256 + 1024)

        # Final detection layer
        x = self.conv_final(x)

        # Reshape output to [batch, S, S, B * (5 + C)]
        batch_size = x.shape[0]
        x = mx.reshape(x, (batch_size, self.S, self.S, self.B * (5 + self.C)))

        if return_features:
            return x, {
                'conv6': conv6_features,
                'conv7': conv7_features,
                'route': route
            }
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
