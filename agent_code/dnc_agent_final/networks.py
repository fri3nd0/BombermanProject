import torch
from torch import nn


NUM_PREV_ACTIONS = 2


class DnCAgentNet(nn.Module):
    """Combination of CNN and MLP to use both multi-channel maps and distinct features for classification of actions
    from a given game state.

    Bomb: https://www.deviantart.com/blackbyte223/art/Fire-Note-410419231
    Avatar: https://startpage.com/av/proxy-image?piurl=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.3DWPMcje4Unfa6MEzNyJXgHaHa%26pid%3DApi&sp=1695117207T49f4550c88754dffd01f8e8ca47c07ec22b5227398ea063a36702fa9a568870d
    """
    def __init__(self, in_channels=2, cnn_dim=256, out_dim=7, feature_dim=1, p_drop=0.3):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(4 + feature_dim + 21 + NUM_PREV_ACTIONS * 6, 512),
            # nn.Linear(feature_dim + 21 + NUM_PREV_ACTIONS * 6, 512),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(512, out_dim),
        )

    def forward(self, coin_view, local_view, features, prev_actions):
        x = torch.concat([coin_view, local_view, features, prev_actions], dim=-1)
        return self.mlp(x)
