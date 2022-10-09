from pong import AgentReinforce
import gym
import random
from collections import deque
import my_pong_utils as pu


def play_episode(env, agent):
    """Let the agent play one episode in the environment."""
    state = env.reset()

    # initialize the deque
    states = deque(maxlen=agent.state_size)
    for _ in range(agent.state_size):
        new_state, _, _, _ = env.step(0)
        new_state = pu.process_pong_frame(new_state)
        states.append(new_state)

    score = 0
    while True:
        # state = pu.process_pong_frame(state)
        action = agent.act(states)
        print(f"{action=}")
        # action = random.choice(range(6))
        next_state, reward, is_done, _ = env.step(action)
        score += reward
        next_state = pu.process_pong_frame(next_state)
        states.append(next_state)
        # agent.step(state, action, next_state, reward)
        if is_done:
            break

    return score


def learn_to_play_pong(env, agent, num_episodes):
    """Agent intereacts with env to learn to play pong."""
    scores = []
    score_window = deque(maxlen=100)
    for episode in range(num_episodes):
        score = play_episode(env, agent)
        scores.append(score)
        score_window.append(score)

    return scores


if __name__ == "__main__":
    env = gym.make("PongDeterministic-v4")
    pong_player = AgentReinforce(num_actions=2, num_states=4)
    scores = learn_to_play_pong(env, pong_player, 5)
    env.close()

    print(f"{scores=}")
