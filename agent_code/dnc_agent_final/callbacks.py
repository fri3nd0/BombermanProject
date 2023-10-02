import torch
import torch.nn.functional as F
import random

import numpy as np

try:
    from .networks import DnCAgentNet, NUM_PREV_ACTIONS
except ImportError:
    from networks import DnCAgentNet, NUM_PREV_ACTIONS

# map actions to numeric class labels
ACTION_MAP = {
    'UP': 0,
    'DOWN': 1,
    'LEFT': 2,
    'RIGHT': 3,
    'BOMB': 4,
    'WAIT': 5
}

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']

# map numeric class labels back to actions
ACTION_MAP_INV = {v: k for k, v in ACTION_MAP.items()}


def get_blast_coords(x, y, arena, power=3):
    """
    Calculates the coordinates that a bombs blast will cover upon explosion.
    Adapted from `items.Bomb.get_blast_coords`.
    :param x: x-coordinate of the bomb
    :param y: y-coordinate of the bomb
    :param arena: field
    :param power: range of the bomb in each direction
    :return: the coordinates that a bombs blast will cover upon explosion
    """
    # adapted from items.Bomb.get_blast_coords
    blast_coords = [(x, y)]

    for i in range(1, power + 1):
        if arena[x + i, y] == -1:
            break
        blast_coords.append((x + i, y))
    for i in range(1, power + 1):
        if arena[x - i, y] == -1:
            break
        blast_coords.append((x - i, y))
    for i in range(1, power + 1):
        if arena[x, y + i] == -1:
            break
        blast_coords.append((x, y + i))
    for i in range(1, power + 1):
        if arena[x, y - i] == -1:
            break
        blast_coords.append((x, y - i))

    return blast_coords


def state_to_features_modularized(game_state, r=4, prev_actions=None):
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :param r: radius of field of view, will be a (2*r-1)x(2*r-1) matrix
    :param prev_actions: Actions taken in the previous steps
    :return: channels, features and a dictionary that describes how actions are mapped to the flipped coordinates.
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    # Create and stack channels that represent features on the map:
    # field (border = 1, free = 0)
    field = game_state['field'] < 0
    # boxes (box = 1, no box = 0)
    box_map = game_state['field'] == 1
    # coins (coin = 1, no coin = 0)
    coin_map = np.zeros_like(game_state['field'])
    if game_state['coins']:
        coins = np.array(game_state['coins']).T
        coin_map[coins[0], coins[1]] = 1
    # explosions + fields that are unsafe when one would try to escape in a straight line away from a bomb
    explosion_map = game_state['explosion_map'] > 0
    for (xb, yb), t in game_state['bombs']:
        blast_coords = get_blast_coords(xb, yb, game_state['field'])
        for (i, j) in [(xb + h, yb) for h in range(-3 + t, 4 - t)] + [(xb, yb + h) for h in range(-3 + t, 4 - t)]:
            if (i, j) in blast_coords and (0 < i < explosion_map.shape[0]) and (0 < j < explosion_map.shape[1]):
                explosion_map[i, j] = 1
    # others (other agent = 1, else 0)
    others_map = np.zeros_like(game_state['field'])
    others = np.array([o[3] for o in game_state['others']]).T
    if len(others) > 0:
        others_map[others[0], others[1]] = 1
    blocked_map = np.clip(field + box_map + explosion_map + others_map, 0.0, 1.0)
    # channels = np.stack([field, box_map, explosion_map, others_map, coin_map], axis=0)
    channels = np.stack([blocked_map, box_map, explosion_map, others_map, coin_map], axis=0)

    # Flip maps so that agent is in top left quadrant
    pos = list(game_state['self'][3])  # current position
    act_remap = {}  # actions that have to be remapped due to the flips

    if pos[0] > 8 and pos[1] <= 8:
        act_remap = {'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
        channels = np.flip(channels, axis=1)
        pos[0] = field.shape[0] - 1 - pos[0]
    elif pos[0] > 8 and pos[1] > 8:
        act_remap = {'LEFT': 'RIGHT', 'RIGHT': 'LEFT', 'UP': 'DOWN', 'DOWN': 'UP'}
        channels = np.flip(channels, axis=1)
        channels = np.flip(channels, axis=2)
        pos[0] = field.shape[0] - 1 - pos[0]
        pos[1] = field.shape[1] - 1 - pos[1]
    elif pos[0] <= 8 and pos[1] > 8:
        act_remap = {'UP': 'DOWN', 'DOWN': 'UP'}
        channels = np.flip(channels, axis=2)
        pos[1] = field.shape[1] - 1 - pos[1]

    act_map = {k: k for k in ACTION_MAP.keys()}  # complete mapping from old actions to actions in flipped space
    for key, value in act_remap.items():
        act_map[key] = value

    # pad the map to make slicing easier
    ext = r - 1  # how far the view extends
    pad = r - 1  # how much padding is needed, (r-1) as we provide one feature further away in each direction
    channels = np.pad(channels, ((0, 0), (pad, pad), (pad, pad)), constant_values=0.0)

    # calculate indices for slicing
    x_l, x_u = (pos[0] + pad - ext, pos[0] + pad + ext + 1)
    y_l, y_u = (pos[1] + pad - ext, pos[1] + pad + ext + 1)

    coin_view = channels[:, x_l:x_u, y_l:y_u]

    # encode the last action as a one-hot vector (zero vector if none was provided)
    prev_actions_one_hot = np.zeros((NUM_PREV_ACTIONS, 6))
    for i, prev_action in enumerate(prev_actions):
        if prev_action in ACTION_MAP:
            prev_actions_one_hot[i, :] = np.eye(len(ACTION_MAP))[ACTION_MAP[act_map[prev_action]]]

    # accumulate hand-crafted features
    features = np.array([float(game_state['self'][2])])

    # direct environment
    local_ext = 1
    local_x_l, local_x_u = (pos[0] + pad - local_ext, pos[0] + pad + local_ext)
    local_y_l, local_y_u = (pos[1] + pad - local_ext, pos[1] + pad + local_ext)

    local_view = np.array([
        channels[:, local_x_l, pos[1] + pad],
        channels[:, local_x_u, pos[1] + pad],
        channels[:, pos[0] + pad, local_y_l],
        channels[:, pos[0] + pad, local_y_u]
    ]).flatten(order='F')
    local_view = np.concatenate([local_view, channels[:1, pos[0] + pad, pos[1] + pad]])

    coin_dirs = np.array([
        np.sum(channels[-1, (local_x_l - 3):(pos[0]+pad), (pos[1] + pad - 3):(pos[1] + pad + 3 + 1)]),
        np.sum(channels[-1, (pos[0]+pad+1):(local_x_u + 3 + 1), (pos[1] + pad - 3):(pos[1] + pad + 3 + 1)]),
        np.sum(channels[-1, (pos[0] + pad - 3):(pos[0] + pad + 3 + 1), (local_y_l - 3):(pos[1]+pad)]),
        np.sum(channels[-1, (pos[0] + pad - 3):(pos[0] + pad + 3 + 1), (pos[0]+pad+1):(local_y_u + 3 + 1)])
    ])

    return (
        {'coin_view': coin_dirs, 'local_view': local_view, 'features': features,
         'prev_actions_one_hot': prev_actions_one_hot},
        act_map)


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # self.model_weights_path = 'models/model_weights_pretrained.pth'
    # self.model_weights_path = 'models/model_weights_enhanced.pth'
    self.model_weights_path = 'models/model_weights.pth'
    # initialize the model
    self.model = DnCAgentNet()
    # load the model weights
    self.model.load_state_dict(torch.load(self.model_weights_path))
    self.model.eval()
    self.num_prev_actions = NUM_PREV_ACTIONS
    self.prev_actions = [None for i in range(self.num_prev_actions)]


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if game_state['step'] == 1:
        # Reload model weights (to keep the agent up-to-date when training is running in parallel)
        if not self.train: self.logger.debug("Load model weights.")
        self.model.load_state_dict(torch.load(self.model_weights_path))
        self.model.eval()

    """
    random_prob = .05
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    """

    features, act_map = state_to_features_modularized(game_state, prev_actions=self.prev_actions[-self.num_prev_actions:])
    act_map_inv = {v: k for k, v in act_map.items()}

    # self.logger.debug("Querying model for action.")
    # sample action
    logits = self.model(
        torch.tensor(features['coin_view'][np.newaxis], dtype=torch.float, requires_grad=False),
        torch.tensor(features['local_view'][np.newaxis], dtype=torch.float, requires_grad=False),
        torch.tensor(features['features'][np.newaxis], dtype=torch.float, requires_grad=False),
        torch.flatten(torch.tensor(features['prev_actions_one_hot'][np.newaxis], dtype=torch.float, requires_grad=False), 1)
    )[0]

    cmd_prob = F.softmax(logits[4:], dim=0).detach().numpy()
    cmd_prob[cmd_prob < 0.02] = 0.0
    cmd_prob = cmd_prob / cmd_prob.sum()
    dir_prob = F.softmax(logits[:4], dim=0).detach().numpy()

    cmd = np.random.choice(range(len(cmd_prob)), p=cmd_prob)

    if not self.train: self.logger.debug(f"move: {cmd_prob[0]:.2f}, bomb: {cmd_prob[1]:.2f}, wait: {cmd_prob[2]:.2f}")

    if self.train:
        if cmd == 1:
            action = ACTION_MAP["BOMB"]
        elif cmd == 2:
            action = ACTION_MAP["WAIT"]
        else:
            action = np.random.choice(range(len(dir_prob)), p=dir_prob)
    else:
        # TODO: Choose good lower limits
        if cmd_prob[2] > 0.15 or cmd == 2:
            action = ACTION_MAP["WAIT"]
            if not self.train: self.logger.debug("WAIT")
        elif cmd_prob[1] > 0.2 or cmd == 1:
            action = ACTION_MAP["BOMB"]
            if not self.train: self.logger.debug("BOMB")
        else:
            dir_prob[dir_prob < 0.2] = 0.0
            dir_prob = dir_prob / dir_prob.sum()

            # action = np.random.choice(range(len(dir_prob)), p=dir_prob)
            p_string = ", ".join([f"{act_map_inv[ACTION_MAP_INV[i]]}={dir_prob[i]:.2f}" for i in range(len(dir_prob))])

            if np.max(dir_prob) < .7:
                action = np.random.choice(range(len(dir_prob)), p=dir_prob)
                if not self.train: self.logger.debug(f"Choice: {action} {act_map_inv[ACTION_MAP_INV[action]]}, {p_string}")
            else:
                action = np.argmax(dir_prob)
                if not self.train: self.logger.debug(f"Argmax: {action} {act_map_inv[ACTION_MAP_INV[action]]}, {p_string}")

    # map action to game orientation
    self.prev_actions.append(act_map_inv[ACTION_MAP_INV[action]])
    return self.prev_actions[-1]
