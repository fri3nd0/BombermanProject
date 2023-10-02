from typing import List

import pickle
import uuid
import os


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.game_states = []
    self.actions = []
    self.events = []


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    # Store old state and chosen action
    self.game_states.append(old_game_state)
    self.actions.append(self_action if self_action is not None else 'WAIT')
    self.events.append(events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    # Store the game states and actions
    self.game_states.append(last_game_state)
    self.actions.append(last_action if last_action is not None else 'WAIT')
    self.events.append(events)
    os.makedirs("../dom_data", exist_ok=True)
    #print("Dir created")
    with open(f"../dom_data/{uuid.uuid1()}.pickle", "wb") as file:
        pickle.dump({
            'game_state': self.game_states,
            'action': self.actions,
            'events': self.events
        }, file)
