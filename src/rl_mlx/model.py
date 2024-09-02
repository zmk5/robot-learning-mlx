import mlx.core as mx
import mlx.nn as nn


class MLP(nn.Module):
    def __init__(
        self, n_layers: int, input_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * n_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = mx.maximum(layer(x), 0.0)
        return self.layers[-1](x)
