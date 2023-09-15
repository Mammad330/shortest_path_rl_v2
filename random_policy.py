import numpy as np
import random

from env import GraphEnv


def main():
    SEED = 0
    np.random.seed(SEED)
    random.seed(SEED)

    num_nodes = 25
    source_node = 20
    destination_node = 24
    bi_prob = 0.3
    num_episodes = 25

    env = GraphEnv(
        num_nodes=num_nodes, destination_node=destination_node,
        bi_prob=bi_prob)

    print(f"env.observation_space.size(): {env.observation_space.shape}")
    print(f"env.action_space.n: {env.action_space.n}")

    for episode in range(num_episodes):
        done = False
        cumm_reward = 0
        path_list = list()

        observation, info = env.reset(starting_node=episode)
        path_list.append(chr(65 + info['current_node']))
        while not done:
            action = np.random.choice(np.argwhere(
                np.array(info['action_mask']) > 0).reshape(-1).tolist())
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            cumm_reward += reward
            path_list.append(chr(65 + info['current_node']))
            # print(f"reward: {reward}")

        print(f"Path: {str([' -> '.join(path_list)])}")
        print(f"Cummulative Reward: {cumm_reward}\n")

    input("Completed training.\nPress Enter to start the final evaluation")


if __name__ == '__main__':
    main()