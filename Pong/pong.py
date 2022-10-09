"""A collection of Deep RL agents."""

import torch
from pong_network import PongNetwork


class AgentReinforce:
    """A REINFORCE Agent"""

    def __init__(self, num_states, num_actions) -> None:
        self.state_size = num_states
        self.action_size = num_actions
        self.network = PongNetwork(num_states, num_actions)

    def step(self):
        pass

    def act(self, state):
        """Returns the action for the state according to policy."""
        with torch.no_grad():
            action = self.network(state)

        return torch.argmax(action)
