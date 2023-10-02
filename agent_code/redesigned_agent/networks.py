import torch
from torch import nn


class AgentNet(nn.Module):
    """Combination of CNN and MLP to use both multi-channel maps and distinct features for classification of actions
    from a given game state.
    """
    def __init__(self, in_channels=5, cnn1_dim=128, cnn2_dim=64, out_dim=6, feature_dim=7, p_drop=0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.last_act = torch.zeros(6)
        self.cnn1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1, 1, 1), padding=0),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(128, cnn1_dim, kernel_size=(in_channels, 3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.cnn2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        self.mlp = nn.Sequential(
            nn.Linear(cnn1_dim + cnn2_dim+self.feature_dim, 1024), #135 or 139
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),

        )
        """
        use softmax for policy learning or no activation for Q learning
        """

    def forward(self, channels, coin_channels, features):
        
        x = self.cnn1(channels)
        y = self.cnn2(coin_channels)
        if torch.sum(features[0, 1:7]) == 0:
            features[0, 1:7] = self.last_act

        #print(features)
        x = torch.concat([x, y, features], dim=-1)

        self.last_act = torch.zeros(6)
        self.last_act[torch.argmax(self.mlp(x)[0, 1:7])+1] = 1

        return self.mlp(x)
