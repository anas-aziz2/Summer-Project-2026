import torch
import torch.nn as nn

from .egnn import EGNNLayer


class ScoreNetwork(nn.Module):

    def __init__(
        self,
        hidden_dim=64,
        num_layers=3
    ):
        super().__init__()

        self.embedding = nn.Linear(1, hidden_dim)

        self.layers = nn.ModuleList(
            [
                EGNNLayer(hidden_dim)
                for _ in range(num_layers)
            ]
        )


    def forward(self, x, log_sigma):

        atom_features = torch.ones(x.shape[0], x.shape[1], 1)

        h = self.embedding(atom_features)

        f_pred = torch.zeros_like(x)

        for layer in self.layers:

            x, h, delta_x = layer(x, h, log_sigma)

            f_pred = f_pred + delta_x

        f_pred = f_pred - f_pred.mean(dim=1, keepdim=True)


        return f_pred