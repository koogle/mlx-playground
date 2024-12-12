import mlx.core as mx
import mlx.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dimensions=784, hidden_dimensions=512, num_classes=10):
        super().__init__()
        self.format_to_input = lambda x: x.reshape(-1, input_dimensions)
        self.linear1 = nn.Linear(input_dimensions, hidden_dimensions)
        self.linear2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.linear3 = nn.Linear(hidden_dimensions, num_classes)
        self.relu = nn.ReLU()

    def __call__(self, x):
        x = self.format_to_input(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x
