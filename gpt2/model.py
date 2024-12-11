from mlx.nn import nn


# Start with a basic seq to seq model
class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x
