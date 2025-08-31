"""
Denoising Diffusion Probabilistic Model (DDPM) for CIFAR-10
Based on the paper: https://arxiv.org/abs/2006.11239
"""

import mlx.core as mx
import mlx.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block with GroupNorm - the core building block of the DDPM U-Net.

    This block serves several critical purposes:
    1. **Feature Transformation**: Two conv layers process features at each resolution
    2. **Time Conditioning**: Injects time information to tell the network the noise level
    3. **Residual Connection**: Helps gradients flow and preserves information
    4. **Normalization**: GroupNorm provides stable training without batch dependencies

    Architecture:
        x → Conv → GroupNorm → ReLU → (+time_emb) → Conv → GroupNorm → (+residual) → ReLU → out

    The residual connection allows the network to learn "refinements" rather than
    full transformations, making training much more stable and effective.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim=None, num_groups=8):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Use GroupNorm instead of BatchNorm for better stability
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)

        self.relu = nn.ReLU()

        # Time embedding projection
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.ReLU(), nn.Linear(time_emb_dim, out_channels)
            )
        else:
            self.time_mlp = None

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def __call__(self, x, time_emb=None):
        """
        Forward pass of the residual block.

        Args:
            x: Input features [batch, height, width, channels]
            time_emb: Time embeddings [batch, time_emb_dim] - tells the network
                     how much noise is in the image (t=0: clean, t=999: pure noise)

        Returns:
            Processed features with same spatial dimensions as input
        """

        h = self.relu(self.norm1(self.conv1(x)))  # Learn spatial patterns

        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            # Broadcast time embedding to all spatial positions
            # Shape: [batch, channels] → [batch, 1, 1, channels]
            h = h + mx.expand_dims(mx.expand_dims(time_emb, 1), 1)

        # Second convolution path: refine features with time knowledge
        h = self.norm2(self.conv2(h))

        # Residual connection: add input to output
        h = self.relu(
            h + self.residual_conv(x)
        )  # residual_conv handles channel mismatch
        return h


class AttentionBlock(nn.Module):
    """Self-attention block"""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)

        # Q, K, V projections
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def __call__(self, x):
        b, h, w, c = x.shape

        # Normalize
        norm_x = self.norm(x)

        # Get Q, K, V
        qkv = self.qkv(norm_x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Reshape for multi-head attention
        # For simplicity, using single head attention here
        q = mx.reshape(q, (b, h * w, c))
        k = mx.reshape(k, (b, h * w, c))
        v = mx.reshape(v, (b, h * w, c))

        # Scaled dot-product attention
        scale = 1.0 / mx.sqrt(mx.array(c, dtype=mx.float32))
        attn = mx.softmax(mx.matmul(q, mx.transpose(k, (0, 2, 1))) * scale, axis=-1)
        out = mx.matmul(attn, v)

        # Reshape back
        out = mx.reshape(out, (b, h, w, c))
        out = self.proj_out(out)

        # Residual connection
        return x + out


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time step embeddings"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, time):
        half_dim = self.dim // 2
        embeddings = mx.log(mx.array(10000)) / (half_dim - 1)
        embeddings = mx.exp(mx.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = mx.concatenate([mx.sin(embeddings), mx.cos(embeddings)], axis=-1)
        return embeddings


class DDPM_UNet(nn.Module):
    """
    U-Net architecture for DDPM
    Designed for CIFAR-10 (32x32 images)
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_multipliers=(1, 2, 2, 2),
        num_res_blocks=2,
        attention_resolutions=(8,),
        dropout=0.1,
    ):
        super().__init__()

        # Time embedding
        time_emb_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling
        self.down_blocks = []
        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_multipliers):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(now_channels, out_channels, time_emb_dim)
                )
                now_channels = out_channels
                channels.append(now_channels)

                # Add attention at specified resolutions
                resolution = 32 // (2**i)
                if resolution in attention_resolutions:
                    self.down_blocks.append(AttentionBlock(now_channels))

            # Downsample (except for the last block)
            if i != len(channel_multipliers) - 1:
                self.down_blocks.append(
                    nn.Conv2d(
                        now_channels, now_channels, kernel_size=3, stride=2, padding=1
                    )
                )
                channels.append(now_channels)

        # Middle blocks
        self.mid_block1 = ResidualBlock(now_channels, now_channels, time_emb_dim)
        self.mid_attn = AttentionBlock(now_channels)
        self.mid_block2 = ResidualBlock(now_channels, now_channels, time_emb_dim)

        # Upsampling
        self.up_blocks = []
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_channels = base_channels * mult

            for j in range(num_res_blocks + 1):
                skip_channels = channels.pop()
                self.up_blocks.append(
                    ResidualBlock(
                        now_channels + skip_channels, out_channels, time_emb_dim
                    )
                )
                now_channels = out_channels

                # Add attention at specified resolutions
                resolution = 32 // (2**i)
                if resolution in attention_resolutions and j == num_res_blocks:
                    self.up_blocks.append(AttentionBlock(now_channels))

            # Upsample (except for the last block)
            if i != 0:
                self.up_blocks.append(
                    nn.ConvTranspose2d(
                        now_channels, now_channels, kernel_size=2, stride=2
                    )
                )

        # Final blocks
        self.norm_out = nn.GroupNorm(8, base_channels)
        self.conv_out = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)

    def __call__(self, x, t):
        """
        Forward pass
        x: [batch_size, height, width, channels] - HWC format
        t: [batch_size] - time steps
        """
        # Time embedding
        t_emb = self.time_mlp(t)

        # Initial conv
        h = self.conv_in(x)

        # Downsampling
        hs = [h]
        for layer in self.down_blocks:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
            hs.append(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Upsampling
        for layer in self.up_blocks:
            if isinstance(layer, ResidualBlock):
                h = mx.concatenate([h, hs.pop()], axis=-1)
                h = layer(h, t_emb)
            else:
                h = layer(h)

        # Final conv
        h = self.norm_out(h)
        h = nn.relu(h)
        h = self.conv_out(h)

        return h
