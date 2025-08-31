import mlx.core as mx
import mlx.nn as nn


class SimpleBlock(nn.Module):
    """Simple residual block without time embedding"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm(out_channels)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()
    
    def __call__(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out


class SimpleUNet(nn.Module):
    """
    Simplified U-Net for CIFAR-10 (32x32 images)
    No time embeddings, just pure denoising
    """
    
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder (downsampling)
        self.enc1 = SimpleBlock(in_channels, 64)  # 32x32
        self.enc2 = SimpleBlock(64, 128, stride=2)  # 16x16
        self.enc3 = SimpleBlock(128, 256, stride=2)  # 8x8
        self.enc4 = SimpleBlock(256, 512, stride=2)  # 4x4
        
        # Bottleneck
        self.bottleneck = SimpleBlock(512, 512)  # 4x4
        
        # Decoder (upsampling)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 8x8
        self.dec4 = SimpleBlock(512, 256)  # 256 + 256 from skip
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 16x16
        self.dec3 = SimpleBlock(256, 128)  # 128 + 128 from skip
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 32x32
        self.dec2 = SimpleBlock(128, 64)  # 64 + 64 from skip
        
        # Final output
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def __call__(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)  # 32x32x64
        e2 = self.enc2(e1)  # 16x16x128
        e3 = self.enc3(e2)  # 8x8x256
        e4 = self.enc4(e3)  # 4x4x512
        
        # Bottleneck
        b = self.bottleneck(e4)  # 4x4x512
        
        # Decoder with skip connections
        d4 = self.up4(b)  # 8x8x256
        d4 = mx.concatenate([d4, e3], axis=-1)  # Concat on channel dim (HWC format)
        d4 = self.dec4(d4)  # 8x8x256
        
        d3 = self.up3(d4)  # 16x16x128
        d3 = mx.concatenate([d3, e2], axis=-1)
        d3 = self.dec3(d3)  # 16x16x128
        
        d2 = self.up2(d3)  # 32x32x64
        d2 = mx.concatenate([d2, e1], axis=-1)
        d2 = self.dec2(d2)  # 32x32x64
        
        # Final output
        out = self.final(d2)  # 32x32x3
        
        return out