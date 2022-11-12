import random
from collections import deque

import config as cfg
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from policy import Policy
from torch import optim

GOAL = 195


def play_episode(env, agent, gamma=1.0):
    """Let the agent play one episode in the environment."""

    state = env.reset()
    rewards = []
    log_probs = []
    while True:
        action, log_prob = agent.act(state)
        log_probs.append(log_prob)
        state, reward, is_done, _ = env.step(action)
        rewards.append(reward)
        if is_done:
            break

    discounts = [gamma**i for i in range(len(rewards) + 1)]
    traj_reward = sum([r * d for (r, d) in zip(rewards, discounts)])

    expected_return = torch.sum(torch.cat(log_probs))
    policy_loss = -expected_return * traj_reward

    return policy_loss, traj_reward


def learn_to_balance(env, policy, num_episodes, print_every=100):
    """Agent intereacts with env to learn to balance the pole."""
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)
    scores = []
    score_window = deque(maxlen=100)
    for episode in range(num_episodes):
        loss, score = play_episode(env, policy)
        scores.append(score)
        score_window.append(score)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % print_every == 0:
            print(f"episode: {episode:5}, avg. score: {np.mean(score_window)}")

        if np.mean(score_window) >= GOAL:
            print(f"solved in {episode} episodes")
            break

    return scores


def balance_pole(env, policy):
    """Balance the pole using the suplied policy."""
    state = env.reset()
    for tsteps in range(10000):
        if isinstance(policy, Policy):
            action, _ = policy.act(state)
        else:
            action = random.choice(range(env.action_space.n))
        state, _, done, _ = env.step(action)
        if done:
            print(f"balanced for {tsteps} time steps.")
            break


def main():
    """Entry point."""
    env = gym.make("CartPole-v1")
    env.reset(seed=200)

    policy = Policy()

    scores = learn_to_balance(env, policy, 2000)

    env = gym.make("CartPole-v1", render_mode="human")
    # Performance of learned policy
    for _ in range(3):
        balance_pole(env, policy)

    # Performance of random policy
    for _ in range(3):
        balance_pole(env, None)

    env.close()

    # Plot training scores
    pd.Series(scores).rolling(100).mean().plot()
    plt.show()


if __name__ == "__main__":
    main()
