import mlx.core as mx
import mlx.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=16384):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def __call__(self, x):
        h = mx.maximum(self.encoder(x), 0)  # ReLU
        return self.decoder(h), h
