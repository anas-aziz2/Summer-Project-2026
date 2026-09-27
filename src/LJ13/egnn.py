import torch
import torch.nn as nn


class MessageMLP(nn.Module):

    def __init__(self, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2*hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

    def forward(self, x):
        return self.net(x)

class FeatureMLP(nn.Module):

    def __init__(self, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)

class CoordMLP(nn.Module):

    def __init__(self, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2*hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        torch.nn.init.xavier_uniform_(
            self.net[-1].weight,
            gain=0.001
        )

        torch.nn.init.zeros_(
            self.net[-1].bias
        )

    def forward(self, x):
        return self.net(x)

class EGNNLayer(nn.Module):

    def __init__(self, hidden_dim=64):
        super().__init__()

        self.message_mlp = MessageMLP(hidden_dim)

        self.feature_mlp = FeatureMLP(hidden_dim)

        self.coord_mlp = CoordMLP(hidden_dim)

    def forward(self, x, h, log_sigma):

        N = x.shape[1]

        diff = x[:,:,None,:] - x[:,None,:,:]
        dist2 = (diff**2).sum(dim=-1, keepdim=True)

        hi = h[:,:,None,:]
        hi = hi.expand(-1,-1,N,-1)

        hj = h[:,None,:,:]
        hj = hj.expand(-1,N,-1,-1)

        log_sigma_pair = log_sigma[:,None,None,:]
        log_sigma_pair = log_sigma_pair.expand(-1,N,N,-1)

        message_input = torch.cat([hi, hj, dist2, log_sigma_pair], dim=-1)

        messages = self.message_mlp(message_input)

        agg_messages = messages.sum(dim=2)

        feature_input = torch.cat([h, agg_messages], dim=-1)

        feature_update = self.feature_mlp(feature_input)

        h_new = h + feature_update

        coord_weights = self.coord_mlp(message_input)

        coord_update = diff * coord_weights

        delta_x = coord_update.sum(dim=2)

        delta_x = delta_x / x.shape[1]

        x_new = x + delta_x

        return x_new, h_new, delta_x