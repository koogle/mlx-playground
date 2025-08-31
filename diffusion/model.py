import mlx.core as mx
import mlx.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """Positional embeddings for the timestep

    Transforms scalar timesteps into high-dimensional representations using
    sinusoidal functions at different frequencies. This allows the model to
    distinguish between different noise levels during the diffusion process.

    WHY THIS IS ESSENTIAL:
    The UNet needs to behave completely differently at different timesteps:
    - At t=1000: Input is pure noise, network must imagine global structure
    - At t=500: Network must denoise while preserving emerging features
    - At t=0: Network must only remove subtle noise, preserve all details

    Without time embeddings, the network would try to apply the same denoising
    operation regardless of noise level, which would fail catastrophically.
    The time embedding acts as a "dial" that tells the network how aggressive
    to be with denoising at each step.

    The encoding uses a geometric progression of frequencies from high to low,
    ensuring both fine-grained and coarse temporal information is captured.
    Each timestep gets a unique, smooth representation where nearby timesteps
    have similar embeddings.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, time):
        half_dim = self.dim // 2

        # Create a geometric progression of frequencies from 1 to 1/10000
        # This gives us wavelengths from 2π to 20000π
        embeddings = mx.log(mx.array(10000)) / (half_dim - 1)
        embeddings = mx.exp(mx.arange(half_dim) * -embeddings)

        # Scale each frequency by the timestep value
        # Shape: [batch_size, half_dim]
        embeddings = time[:, None] * embeddings[None, :]

        # Apply sine to first half and cosine to second half
        # This creates orthogonal components for each frequency
        # Final shape: [batch_size, dim]
        embeddings = mx.concatenate([mx.sin(embeddings), mx.cos(embeddings)], axis=-1)
        return embeddings


class Block(nn.Module):
    """Basic convolutional block with residual connection"""

    def __init__(self, in_ch, out_ch, time_emb_dim=None, up=False):
        super().__init__()

        # Time embedding projection
        self.time_mlp = (
            nn.Sequential(nn.ReLU(), nn.Linear(time_emb_dim, out_ch))
            if time_emb_dim is not None
            else None
        )

        # Explicitly set shapes for convolutions
        if up:
            self.conv1 = nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            w_shape = (in_ch, out_ch, 3, 3)  # Shape for ConvTranspose2d
            self.conv1.weight = mx.random.normal(w_shape) * 0.02

            self.transform = nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            self.transform.weight = mx.random.normal(w_shape) * 0.02
        else:
            self.conv1 = nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                padding=1,
            )
            w_shape = (out_ch, in_ch, 3, 3)  # Shape for Conv2d
            self.conv1.weight = mx.random.normal(w_shape) * 0.02

            if in_ch != out_ch:
                self.transform = nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=1,
                )
                w_shape = (out_ch, in_ch, 1, 1)
                self.transform.weight = mx.random.normal(w_shape) * 0.02
            else:
                self.transform = nn.Identity()

        self.conv2 = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=3,
            padding=1,
        )
        w_shape = (out_ch, out_ch, 3, 3)
        self.conv2.weight = mx.random.normal(w_shape) * 0.02
        self.bnorm1 = nn.BatchNorm(out_ch)
        self.bnorm2 = nn.BatchNorm(out_ch)
        self.relu = nn.ReLU()

    def __call__(self, x, t=None):
        h = self.bnorm1(self.relu(self.conv1(x)))

        if self.time_mlp is not None and t is not None:
            time_emb = self.relu(self.time_mlp(t))
            h = h + time_emb[..., None, None]

        h = self.bnorm2(self.relu(self.conv2(h)))
        return h + self.transform(x)


class UNet(nn.Module):
    """
    U-Net architecture for diffusion models.
    """

    def __init__(
        self,
        in_channels=3,
        model_channels=32,
        out_channels=3,
        num_res_blocks=1,
        attention_levels=[2],
        channel_mult=(1, 2, 4),
        time_emb_dim=None,
    ):
        super().__init__()
        self.out_channels = out_channels  # Store as instance variable

        # Time embeddings
        self.time_mlp = (
            nn.Sequential(
                SinusoidalPositionEmbeddings(model_channels),
                nn.Linear(model_channels, model_channels * 4),
                nn.ReLU(),
                nn.Linear(model_channels * 4, model_channels * 4),
            )
            if time_emb_dim is not None
            else None
        )

        # Initial convolution - need to transpose the weight shape
        self.conv_in = nn.Conv2d(
            in_channels=in_channels,  # 3
            out_channels=model_channels,  # 64
            kernel_size=3,
            padding=1,
        )
        # Manually fix the weight shape
        w_shape = (model_channels, in_channels, 3, 3)  # (out, in, k, k)
        self.conv_in.weight = mx.random.normal(w_shape) * 0.02

        # Downsampling
        self.downs = []
        channels = [model_channels]  # Track channels for skip connections
        now_channels = model_channels

        for level, mult in enumerate(channel_mult):
            out_channels = model_channels * mult

            # Add res blocks
            for _ in range(num_res_blocks):
                block = Block(
                    in_ch=now_channels,
                    out_ch=out_channels,
                    time_emb_dim=(
                        model_channels * 4 if time_emb_dim is not None else None
                    ),
                )
                self.downs.append(block)
                now_channels = out_channels
                channels.append(now_channels)

            # Add downsampling
            if level != len(channel_mult) - 1:
                self.downs.append(nn.MaxPool2d(2))
                channels.append(now_channels)

        # Middle block
        self.mid = Block(
            in_ch=now_channels,
            out_ch=now_channels,
            time_emb_dim=model_channels * 4 if time_emb_dim is not None else None,
        )

        # Upsampling
        self.ups = []
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_channels = model_channels * mult

            # Add res blocks with skip connections
            for i in range(num_res_blocks + 1):
                # Input channels include skip connection
                skip_channels = channels.pop()
                in_channels = skip_channels + now_channels

                # Create upsampling block
                block = Block(
                    in_ch=in_channels,
                    out_ch=out_channels,
                    time_emb_dim=(
                        model_channels * 4 if time_emb_dim is not None else None
                    ),
                    up=True if i == num_res_blocks else False,
                )
                self.ups.append(block)
                now_channels = out_channels

        # Final output convolution
        self.conv_out = nn.Sequential(
            nn.BatchNorm(now_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=now_channels,
                out_channels=self.out_channels,  # Use self.out_channels
                kernel_size=3,
                padding=1,
            ),
        )

    def __call__(self, x, time=None):
        """
        Forward pass of UNet
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            time: Time embeddings tensor of shape [batch_size, 1]
        """
        # Process time embeddings if provided
        if self.time_mlp is not None and time is not None:
            # Ensure time has correct shape
            if len(time.shape) == 1:
                time = mx.expand_dims(time, axis=-1)
            t = self.time_mlp(time)
        else:
            t = None

        # Initial conv
        h = self.conv_in(x)
        hs = [h]

        # Downsampling
        for layer in self.downs:
            if isinstance(layer, Block):
                h = layer(h, t)
            else:
                h = layer(h)
            hs.append(h)

        # Middle
        h = self.mid(h, t)

        # Upsampling
        for layer in self.ups:
            if isinstance(layer, Block):
                # Pop the skip connection from the stack
                skip = hs.pop()
                # Concatenate along the channel dimension
                h = mx.concatenate([h, skip], axis=1)
                h = layer(h, t)

        return self.conv_out(h)
