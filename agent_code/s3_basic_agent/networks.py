import torch
from torch import nn


class AgentNet(nn.Module):
    """Combination of CNN and MLP to use both multi-channel maps and distinct features for classification of actions
    from a given game state.
    """
    def __init__(self, in_channels=5, cnn_dim=128, out_dim=6, feature_dim=7, p_drop=0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.last_act = torch.zeros(6)
        self.cnn = nn.Sequential(

            nn.Conv3d(1, 32, kernel_size=(1, 1, 1), padding=0),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(128, cnn_dim, kernel_size=(in_channels, 3, 3), stride=1, padding=0),
            # nn.ReLU(),
            nn.Flatten(),
            # nn.Dropout(p_drop),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Dropout(p_drop),
            # nn.Linear(256, cnn_dim),
            # nn.ReLU(),
            # nn.Dropout(p_drop),
        )

        self.mlp = nn.Sequential(
            nn.Linear(cnn_dim + self.feature_dim, 1024), #135 or 139
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, out_dim),

        )
        """
        use softmax for policy learning or no activation for Q learning
        """

    def forward(self, channels, features):
        
        x = self.cnn(channels)
        if torch.sum(features[0, 1:7]) == 0:
            features[0, 1:7] = self.last_act

        #print(features)
        x = torch.concat([x, features], dim=-1)

        self.last_act = torch.zeros(6)
        self.last_act[torch.argmax(self.mlp(x)[0, 1:7])+1] = 1

        return self.mlp(x)
