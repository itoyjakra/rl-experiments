import random
from collections import deque

import gym
import my_pong_utils as pu
import numpy as np
import torch
from pong import AgentReinforce
from torch import optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def play_episode(env, agent, gamma=1.0):
    """Let the agent play one episode in the environment."""
    state = env.reset()

    # initialize the deque
    states = deque(maxlen=agent.state_size)
    for _ in range(agent.state_size):
        new_state, _, _, _ = env.step(0)
        new_state = pu.process_pong_frame(new_state)
        states.append(new_state)

    rewards = []
    log_probs = []
    score = 0
    counter = 0
    max_steps = 500
    while True:
        # state = pu.process_pong_frame(state)
        # print(len(states))
        # print(states[0].shape)
        states_array = np.array(states)
        states_array = torch.from_numpy(states_array).float().unsqueeze(0).to(device)
        action, log_prob = agent.act(states_array)
        # print(f"{action=}, {log_prob=}")
        x = log_prob.squeeze(0).numpy()
        log_probs.append(log_prob.reshape(1))
        next_state, reward, is_done, _ = env.step(action)
        rewards.append(reward)
        score += reward
        next_state = pu.process_pong_frame(next_state)
        states.append(next_state)
        counter += 1
        if is_done or counter > max_steps:
            break

    discounts = [gamma**i for i in range(len(rewards) + 1)]
    traj_reward = sum([r * d for (r, d) in zip(rewards, discounts)])

    print(f"{counter=}")
    print(f"{traj_reward=}")

    expected_return = torch.sum(torch.cat(log_probs))
    policy_loss = -expected_return * traj_reward

    return policy_loss, traj_reward


def learn_to_play_pong(env, agent, num_episodes):
    """Agent intereacts with env to learn to play pong."""
    optimizer = optim.Adam(agent.network.parameters(), lr=1e-3)

    scores = []
    score_window = deque(maxlen=100)
    for episode in range(num_episodes):
        loss, reward = play_episode(env, agent)
        scores.append(reward)
        score_window.append(reward)

        optimizer.zero_grad()
        loss.requires_grad = True
        loss.backward()
        optimizer.step()

    return scores


if __name__ == "__main__":
    env = gym.make("PongDeterministic-v4")
    pong_player = AgentReinforce(num_actions=2, num_states=4)
    scores = learn_to_play_pong(env, pong_player, 1000)
    env.close()

    print(f"{scores=}")
