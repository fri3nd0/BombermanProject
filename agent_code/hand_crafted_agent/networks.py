import torch
from torch import nn


class AgentNet(nn.Module):
    def __init__(self, in_channels=2, cnn_dim=256, out_dim=6, feature_dim=7, p_drop=0.3):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(4 + feature_dim + 21, 512),
            # nn.Linear(feature_dim + 21 + NUM_PREV_ACTIONS * 6, 512),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(512, out_dim),
        )

    def forward(self, coin_view, local_view, features):
        x = torch.concat([coin_view, local_view, features], dim=-1)
        return self.mlp(x)
