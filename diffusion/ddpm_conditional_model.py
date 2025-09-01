"""
Conditional DDPM model with CLIP-style embeddings for class-conditional generation
Supports both text and class label conditioning
"""

import mlx.core as mx
import mlx.nn as nn


class ClassEmbedding(nn.Module):
    """
    Class embedding layer that creates learnable embeddings for each class
    Similar to CLIP's approach but simpler for class labels
    """

    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Learnable embedding matrix for each class
        self.embeddings = nn.Embedding(num_classes, embedding_dim)

        # Additional projection layers for richer representations
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def __call__(self, class_labels):
        """
        Args:
            class_labels: [batch_size] tensor of class indices
        Returns:
            [batch_size, embedding_dim] class embeddings
        """
        # Get base embeddings
        embeddings = self.projection(self.embeddings(class_labels))

        return embeddings


class ConditionalResidualBlock(nn.Module):
    """
    Residual block with both time and class conditioning
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim=None,
        class_emb_dim=None,
        num_groups=8,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Use GroupNorm for stability
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

        # Class embedding projection
        if class_emb_dim is not None:
            self.class_mlp = nn.Sequential(
                nn.ReLU(), nn.Linear(class_emb_dim, out_channels)
            )
        else:
            self.class_mlp = None

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def __call__(self, x, time_emb=None, class_emb=None):
        """
        Forward pass with time and class conditioning

        Args:
            x: Input features [batch, height, width, channels]
            time_emb: Time embeddings [batch, time_emb_dim]
            class_emb: Class embeddings [batch, class_emb_dim]
        """
        h = self.relu(self.norm1(self.conv1(x)))

        # Add time embedding
        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            h = h + mx.expand_dims(mx.expand_dims(time_emb, 1), 1)

        # Add class embedding
        if self.class_mlp is not None and class_emb is not None:
            class_emb = self.class_mlp(class_emb)
            h = h + mx.expand_dims(mx.expand_dims(class_emb, 1), 1)

        h = self.norm2(self.conv2(h))
        h = self.relu(h + self.residual_conv(x))

        return h


class ConditionalAttentionBlock(nn.Module):
    """
    Self-attention block with optional cross-attention to class embeddings
    """

    def __init__(self, channels, num_heads=4, class_emb_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)

        # Self-attention components
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

        # Cross-attention to class embeddings (optional)
        if class_emb_dim is not None:
            self.cross_attention = True
            self.norm_class = nn.LayerNorm(class_emb_dim)
            self.class_kv = nn.Linear(class_emb_dim, channels * 2)
        else:
            self.cross_attention = False

    def __call__(self, x, class_emb=None):
        b, h, w, c = x.shape

        # Self-attention
        norm_x = self.norm(x)
        qkv = self.qkv(norm_x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Reshape for attention
        q = mx.reshape(q, (b, h * w, c))
        k = mx.reshape(k, (b, h * w, c))
        v = mx.reshape(v, (b, h * w, c))

        # Scaled dot-product attention
        scale = 1.0 / mx.sqrt(mx.array(c, dtype=mx.float32))
        attn = mx.softmax(mx.matmul(q, mx.transpose(k, (0, 2, 1))) * scale, axis=-1)
        out = mx.matmul(attn, v)

        # Optional cross-attention to class embeddings
        if self.cross_attention and class_emb is not None:
            # Normalize class embeddings
            class_emb = self.norm_class(class_emb)

            # Get key and value from class embeddings
            class_kv = self.class_kv(class_emb)
            class_k, class_v = mx.split(class_kv, 2, axis=-1)

            # Add class dimension for attention
            class_k = mx.expand_dims(class_k, 1)  # [b, 1, channels]
            class_v = mx.expand_dims(class_v, 1)  # [b, 1, channels]

            # Cross-attention
            cross_attn = mx.softmax(
                mx.matmul(q, mx.transpose(class_k, (0, 2, 1))) * scale, axis=-1
            )
            cross_out = mx.matmul(cross_attn, class_v)

            # Combine self and cross attention
            out = out + 0.1 * cross_out  # Small weight for cross-attention initially

        # Reshape back and project
        out = mx.reshape(out, (b, h, w, c))
        out = self.proj_out(out)

        # Residual connection
        return x + out


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for timesteps"""

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


class ConditionalDDPM_UNet(nn.Module):
    """
    Conditional U-Net for class-conditional DDPM
    Incorporates class information at multiple scales
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
        num_classes=10,  # Number of classes for conditioning
        class_emb_dim=128,  # Dimension of class embeddings
        use_cross_attention=False,  # Whether to use cross-attention to class
    ):
        super().__init__()

        self.num_classes = num_classes
        self.class_emb_dim = class_emb_dim

        # Class embedding layer
        self.class_embedding = ClassEmbedding(
            num_classes + 1, class_emb_dim
        )  # +1 for unconditional

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

        # Downsampling path
        self.down_blocks = []
        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_multipliers):
            out_channels = base_channels * mult

            for j in range(num_res_blocks):
                self.down_blocks.append(
                    ConditionalResidualBlock(
                        now_channels, out_channels, time_emb_dim, class_emb_dim
                    )
                )
                now_channels = out_channels
                channels.append(now_channels)

                # Add attention at specified resolutions
                if 32 // (2**i) in attention_resolutions:
                    self.down_blocks.append(
                        ConditionalAttentionBlock(
                            now_channels,
                            class_emb_dim=(
                                class_emb_dim if use_cross_attention else None
                            ),
                        )
                    )
                    channels.append(now_channels)

            # Downsample (except for the last block)
            if i != len(channel_multipliers) - 1:
                self.down_blocks.append(
                    nn.Conv2d(
                        now_channels, now_channels, kernel_size=3, stride=2, padding=1
                    )
                )
                channels.append(now_channels)

        # Middle blocks with attention
        self.mid_block1 = ConditionalResidualBlock(
            now_channels, now_channels, time_emb_dim, class_emb_dim
        )
        self.mid_attention = ConditionalAttentionBlock(
            now_channels, class_emb_dim=class_emb_dim if use_cross_attention else None
        )
        self.mid_block2 = ConditionalResidualBlock(
            now_channels, now_channels, time_emb_dim, class_emb_dim
        )

        # Upsampling path
        self.up_blocks = []
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_channels = base_channels * mult

            # Upsample first (except for the last block)
            if i != len(channel_multipliers) - 1:
                self.up_blocks.append(
                    nn.ConvTranspose2d(
                        now_channels, now_channels, kernel_size=2, stride=2
                    )
                )

            for j in range(num_res_blocks + 1):
                skip_channels = channels.pop()
                self.up_blocks.append(
                    ConditionalResidualBlock(
                        now_channels + skip_channels,
                        out_channels,
                        time_emb_dim,
                        class_emb_dim,
                    )
                )
                now_channels = out_channels

                # Add attention at specified resolutions
                if 32 // (2**i) in attention_resolutions and j == num_res_blocks:
                    self.up_blocks.append(
                        ConditionalAttentionBlock(
                            now_channels,
                            class_emb_dim=(
                                class_emb_dim if use_cross_attention else None
                            ),
                        )
                    )

        # Final layers
        self.norm_out = nn.GroupNorm(8, base_channels)
        self.conv_out = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)

    def __call__(self, x, t, class_labels=None, unconditional_prob=0.1):
        """
        Forward pass with class conditioning

        Args:
            x: [batch_size, height, width, channels] - Input images (HWC format)
            t: [batch_size] - Timesteps
            class_labels: [batch_size] - Class labels (None for unconditional)
            unconditional_prob: Probability of dropping class info (for classifier-free guidance)
        """
        batch_size = x.shape[0]

        # Handle class embeddings
        if class_labels is None:
            # Use unconditional token (last index)
            class_labels = mx.ones(batch_size, dtype=mx.int32) * self.num_classes
        else:
            # Randomly drop class labels for classifier-free guidance training
            if unconditional_prob > 0:
                mask = mx.random.uniform(shape=(batch_size,)) < unconditional_prob
                class_labels = mx.where(
                    mask,
                    mx.ones_like(class_labels)
                    * self.num_classes,  # Unconditional token
                    class_labels,
                )

        # Get embeddings
        class_emb = self.class_embedding(class_labels)
        t_emb = self.time_mlp(t)

        # Initial conv
        h = self.conv_in(x)

        # Downsampling with skip connections
        hs = [h]
        for layer in self.down_blocks:
            if isinstance(layer, ConditionalResidualBlock):
                h = layer(h, t_emb, class_emb)
                hs.append(h)
            elif isinstance(layer, ConditionalAttentionBlock):
                h = layer(h, class_emb)
                hs.append(h)
            else:  # Conv2d downsampling
                h = layer(h)
                hs.append(h)

        # Middle
        h = self.mid_block1(h, t_emb, class_emb)
        h = self.mid_attention(h, class_emb)
        h = self.mid_block2(h, t_emb, class_emb)

        # Upsampling with skip connections
        for layer in self.up_blocks:
            if isinstance(layer, nn.ConvTranspose2d):
                # Upsample first
                h = layer(h)
            elif isinstance(layer, ConditionalResidualBlock):
                # Then concatenate with skip connection
                skip = hs.pop()
                h = mx.concatenate([h, skip], axis=-1)
                h = layer(h, t_emb, class_emb)
            elif isinstance(layer, ConditionalAttentionBlock):
                h = layer(h, class_emb)

        # Final conv
        h = self.norm_out(h)
        h = nn.relu(h)
        h = self.conv_out(h)

        return h
