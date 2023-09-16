import numpy as np
import random

from env import GraphEnv


def main():
    # Set random generator seed
    np.random.seed(SEED)
    random.seed(SEED)

    # ASCII offset for converting node numbers to alphabets for printing
    ascii_offset = 65 if NUM_NODES <= 60 else 21 if NUM_NODES <= 100 else None

    # Create the environment
    env = GraphEnv(
        num_nodes=NUM_NODES, destination_node=DESTINATION_NODE,
        edge_prob=EDGE_PROB, trans_prob_low=TRANS_PROB_LOW, lambda_=LAMBDA,
        trans_prob_high=TRANS_PROB_HIGH)

    print(f"env.observation_space.size(): {env.observation_space.shape}")
    print(f"env.action_space.n: {env.action_space.n}")

    # Create a list of valid starting nodes (all nodes except the destination
    # node)
    starting_nodes = list(range(NUM_NODES))
    starting_nodes.remove(DESTINATION_NODE)

    # Iterate over all valid starting nodes
    for node in starting_nodes:
        path_list = list()
        action_list = list()
        cumm_reward = 0
        done = False

        observation, info = env.reset(starting_node=node)
        path_list.append(str(info['current_node']) if ascii_offset is None else
                         chr(ascii_offset + info['current_node']))
        while not done:
            action = np.random.choice(np.argwhere(
                np.array(info['action_mask']) > 0).reshape(-1).tolist())
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            cumm_reward += reward
            path_list.append(
                str(info['current_node']) if ascii_offset is None else
                chr(ascii_offset + info['current_node']))
            action_list.append(str(action) if ascii_offset is None else
                               chr(ascii_offset + action))

        # print(f"Path: {str([' -> '.join(path_list)])}")
        path_string = path_list[0]
        for path, action in zip(path_list[1:], action_list):
            path_string += f" ({action}) -> {path}"
        print(f"Path: {path_string}")
        print(f"Cumulative Reward: {cumm_reward}\n")

    input("Completed training.\nPress Enter to start the final evaluation")


if __name__ == '__main__':
    # Hyperparameters for graph generation
    SEED = 0
    NUM_NODES = 25
    DESTINATION_NODE = 20
    EDGE_PROB = 0.3
    TRANS_PROB_LOW = 0.5
    TRANS_PROB_HIGH = 1.0
    LAMBDA = 0.5

    main()