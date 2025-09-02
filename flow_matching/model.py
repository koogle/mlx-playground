import mlx.core as mx
import mlx.nn as nn
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, time):
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = mx.exp(mx.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = mx.concatenate([mx.sin(embeddings), mx.cos(embeddings)], axis=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        else:
            self.time_mlp = None

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)

    def __call__(self, x, t=None):
        h = self.conv1(x)
        h = self.norm1(h)

        if self.time_mlp is not None and t is not None:
            time_emb = self.time_mlp(t)
            h = h + time_emb[:, :, None, None]

        h = nn.relu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = nn.relu(h)

        if self.residual_conv is not None:
            x = self.residual_conv(x)

        return h + x


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def __call__(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.transpose(1, 0, 2, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = (C // self.num_heads) ** -0.5
        attn = mx.matmul(q.transpose(0, 1, 3, 2), k) * scale
        attn = mx.softmax(attn, axis=-1)

        out = mx.matmul(attn, v.transpose(0, 1, 3, 2))
        out = out.transpose(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj(out)

        return out + x


class FlowMatchModel(nn.Module):
    def __init__(
        self,
        input_channels=3,
        hidden_channels=128,
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        channel_mult=[1, 2, 4, 8],
        num_heads=4,
        time_emb_dim=256,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_heads = num_heads

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_channels),
            nn.Linear(hidden_channels, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial convolution
        self.conv_in = nn.Conv2d(
            input_channels, hidden_channels, kernel_size=3, padding=1
        )

        # Downsampling blocks
        self.down_blocks = []
        channels = [hidden_channels]
        now_channels = hidden_channels

        for i, mult in enumerate(channel_mult):
            out_channels = hidden_channels * mult

            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(now_channels, out_channels, time_emb_dim)
                )
                now_channels = out_channels
                channels.append(now_channels)

            if i != len(channel_mult) - 1:
                self.down_blocks.append(
                    nn.Conv2d(
                        now_channels, now_channels, kernel_size=3, stride=2, padding=1
                    )
                )
                channels.append(now_channels)

        # Middle blocks
        self.middle_blocks = [
            ResidualBlock(now_channels, now_channels, time_emb_dim),
            AttentionBlock(now_channels, num_heads),
            ResidualBlock(now_channels, now_channels, time_emb_dim),
        ]

        # Upsampling blocks
        self.up_blocks = []
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_channels = hidden_channels * mult

            for j in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResidualBlock(
                        now_channels + channels.pop(), out_channels, time_emb_dim
                    )
                )
                now_channels = out_channels

            if i != 0:
                self.up_blocks.append(
                    nn.ConvTranspose2d(
                        now_channels, now_channels, kernel_size=4, stride=2, padding=1
                    )
                )

        # Final layers
        self.norm_out = nn.GroupNorm(8, now_channels)
        self.conv_out = nn.Conv2d(
            now_channels, input_channels, kernel_size=3, padding=1
        )

    def __call__(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)

        # Initial conv
        h = self.conv_in(x)

        # Downsampling
        hs = [h]
        for block in self.down_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, t_emb)
            else:
                h = block(h)
            hs.append(h)

        # Middle
        for block in self.middle_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, t_emb)
            else:
                h = block(h)

        # Upsampling
        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                h = mx.concatenate([h, hs.pop()], axis=1)
                h = block(h, t_emb)
            else:
                h = block(h)

        # Final layers
        h = self.norm_out(h)
        h = nn.silu(h)
        h = self.conv_out(h)

        return h
