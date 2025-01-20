import mlx.core as mx
import mlx.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """Positional embeddings for the timestep"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = mx.log(mx.array(10000)) / (half_dim - 1)
        embeddings = mx.exp(mx.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = mx.concatenate([mx.sin(embeddings), mx.cos(embeddings)], axis=-1)
        return embeddings


class Block(nn.Module):
    """Basic convolutional block with residual connection"""

    def __init__(self, in_ch, out_ch, time_emb_dim=None, up=False):
        super().__init__()
        self.time_mlp = (
            nn.Sequential(nn.ReLU(), nn.Linear(time_emb_dim, out_ch))
            if time_emb_dim is not None
            else None
        )

        if up:
            self.conv1 = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1)
            self.transform = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = (
                nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            )

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
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
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_levels=[2],
        channel_mult=(1, 2, 4, 8),
        time_emb_dim=None,
    ):
        super().__init__()

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

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        # Downsampling
        self.downs = []
        channels = [model_channels]
        now_channels = model_channels
        for level, mult in enumerate(channel_mult):
            out_channels = model_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(
                    Block(now_channels, out_channels, time_emb_dim=model_channels * 4)
                )
                now_channels = out_channels
                channels.append(now_channels)
            if level != len(channel_mult) - 1:
                self.downs.append(nn.MaxPool2d(2))
                channels.append(now_channels)

        # Middle
        self.mid = Block(now_channels, now_channels, time_emb_dim=model_channels * 4)

        # Upsampling
        self.ups = []
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_channels = model_channels * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(
                    Block(
                        channels.pop() + now_channels,
                        out_channels,
                        time_emb_dim=model_channels * 4,
                        up=True if _ == num_res_blocks else False,
                    )
                )
                now_channels = out_channels

        self.conv_out = nn.Sequential(
            nn.BatchNorm(now_channels),
            nn.ReLU(),
            nn.Conv2d(now_channels, out_channels, 3, padding=1),
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
                h = layer(mx.concatenate([h, hs.pop()], axis=1), t)

        return self.conv_out(h)
