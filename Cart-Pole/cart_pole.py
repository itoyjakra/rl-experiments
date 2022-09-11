import gym
import torch
import torch.optim as optim
import numpy as np
from policy import Policy
from collections import deque
import matplotlib.pyplot as plt
import config as cfg
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reinforce(env, policy, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma**i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print(f"episode: {i_episode:5}, avg. score: {np.mean(scores_deque)}")

        if np.mean(scores_deque) >= 195.0:
            print(f"solved in {i_episode} episodes")
            break

    return scores


def main():
    """Entry point."""
    env = gym.make("CartPole-v1")
    env.reset(seed=200)
    policy = Policy()
    scores = reinforce(env, policy, n_episodes=4000)

    pd.Series(scores).rolling(100).mean().plot()
    plt.show()


if __name__ == "__main__":
    main()
