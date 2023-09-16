import numpy as np
import os
import time
import json
import random
import argparse

import torch
import torch.nn as nn

from datetime import datetime
from shutil import rmtree
from typing import Mapping

from env import GraphEnv
from dql import DoubleDQL
from utils import bellman_ford


def save_learning_params(max_train_steps: int
                         ) -> Mapping[str, bool | int | float]:
    """
    Saves the training parameters in separate file for reference
    Parameters
    ----------
    max_train_steps: int
        Maximum number of training steps

    Returns
    -------
    dict:
        A dictionary of training parameters
    """
    # Initialize the dictionary
    learning_params = dict()

    # Environment
    # -----------
    learning_params['seed'] = SEED
    learning_params['num_nodes'] = NUM_NODES
    learning_params['destination_node'] = DESTINATION_NODE
    learning_params['edge_prob'] = EDGE_PROB
    learning_params['trans_prob_low'] = TRANS_PROB_LOW
    learning_params['trans_prob_high'] = TRANS_PROB_HIGH
    learning_params['lambda'] = LAMBDA

    # Adam
    # ----
    learning_params['lr'] = LR
    learning_params['beta_1'] = BETA_1
    learning_params['beta_2'] = BETA_2

    # DQL
    # -----
    learning_params['gamma'] = GAMMA
    learning_params['rho'] = EPSILON
    learning_params['epsilon_decay'] = EPSILON_DECAY
    learning_params['epsilon_min'] = EPSILON_MIN
    learning_params['replay_mem_size'] = REPLAY_MEM_SIZE
    learning_params['initial_period'] = INITIAL_PERIOD
    learning_params['main_update_period'] = MAIN_UPDATE_PERIOD
    learning_params['target_update_period'] = TARGET_UPDATE_PERIOD

    # NN
    # ----
    learning_params['hl1_size'] = HL1_SIZE
    learning_params['hl2_size'] = HL2_SIZE
    learning_params['batch_size'] = BATCH_SIZE

    # Training
    # --------
    learning_params['max_train_steps'] = max_train_steps
    learning_params['eval_freq'] = EVALUATION_FREQUENCY

    return learning_params


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random generator seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Create a timestamp directory to save model, parameter and log files
    train_path = \
        ('training/DQL/' + str(datetime.now().date()) + '_' +
         str(datetime.now().hour).zfill(2) + '-' +
         str(datetime.now().minute).zfill(2) + '/')

    # Delete if a directory with the same name already exists
    if os.path.exists(train_path):
        rmtree(train_path)

    # Create empty directories for saving model, parameter and log files
    os.makedirs(train_path)
    os.makedirs(train_path + 'plots')
    os.makedirs(train_path + 'learning')
    os.makedirs(train_path + 'models')

    # Create the environment
    env = GraphEnv(
        num_nodes=NUM_NODES, destination_node=DESTINATION_NODE,
        edge_prob=EDGE_PROB, trans_prob_low=TRANS_PROB_LOW, lambda_=LAMBDA,
        trans_prob_high=TRANS_PROB_HIGH)

    # If the number of nodes is less than 200, using Bellman-Ford algorithm,
    # calculate the expected average reward for reaching the destination node
    # starting from each valid starting node
    if NUM_NODES <= 200:
        start_time = time.time()
        graph_copy = np.copy(env.graph)
        distances, shortest_paths = bellman_ford(
            graph_copy, env.trans_prob, NUM_NODES, DESTINATION_NODE,
            env.possible_starting_nodes, LAMBDA)
        end_time = time.time()
        print(f"Time taken for Bellman-Ford: {end_time - start_time:.2f} sec")

        # The expected average reward is the mean of the difference between the
        # maximum possible reward and the distance from each valid starting
        # node to the destination node
        expected_avg_reward = np.mean((NUM_NODES * 10) - np.array(distances))
        print(f"expected_avg_reward: {np.round(expected_avg_reward, 4)}")
    else:
        # If the number of nodes is greater than 200, the expected average
        # reward is not calculated as the time complexity of Bellman-Ford
        # algorithm is O(V*E) and the number of edges is O(V^2), so
        # O(V*E) = O(V^3) and for large number of nodes (V), i.e. > 200,
        # O(V^3) is very large. So, for large number of nodes, the expected
        # average reward is not calculated as it is not required for training
        # the agent
        expected_avg_reward = None

    # expected_avg_reward = 4985.52795  # Pre-calculated for 500 nodes

    print(f"env.observation_space.size(): {env.observation_space.shape}")
    print(f"env.action_space: {env.action_space.n}")

    # Save the learning parameters for reference
    learning_params = save_learning_params(args.max_train_steps)

    # Dump learning params to file
    with open(train_path + 'learning/params.dat', 'w') as jf:
        json.dump(learning_params, jf, indent=4)

    # Loss function for optimization - Mean Squared Error loss
    mse_loss = nn.MSELoss()

    # The DoubleDQL class object
    dqn = DoubleDQL(
        train_env=env, loss_fn=mse_loss, gamma=GAMMA, lr=LR, epsilon=EPSILON,
        replay_mem_size=REPLAY_MEM_SIZE, hl1_size=HL1_SIZE, hl2_size=HL2_SIZE,
        device=device)

    # Train the agent
    dqn.train(
        training_steps=args.max_train_steps, init_training_period=INITIAL_PERIOD,
        main_update_period=MAIN_UPDATE_PERIOD, batch_size=BATCH_SIZE,
        target_update_period=TARGET_UPDATE_PERIOD, show_plot=args.plot,
        evaluation_freq=EVALUATION_FREQUENCY, baseline=expected_avg_reward,
        path=train_path)


if __name__ == '__main__':
    # Environment
    # -----------
    SEED = 0
    NUM_NODES = 25
    DESTINATION_NODE = 20
    EDGE_PROB = 0.3
    TRANS_PROB_LOW = 0.1
    TRANS_PROB_HIGH = 0.2
    LAMBDA = 0.5

    # Adam
    # ------
    LR = 0.001
    BETA_1 = 0.9
    BETA_2 = 0.999

    # TDL
    # -----
    GAMMA = 0.99
    EPSILON = 1.0
    EPSILON_DECAY = 0.99999
    EPSILON_MIN = 0.01

    # NN
    # ----
    HL1_SIZE = 256
    HL2_SIZE = 256
    BATCH_SIZE = 256

    # DQL
    # -----
    REPLAY_MEM_SIZE = 500_000
    INITIAL_PERIOD = 5000
    MAIN_UPDATE_PERIOD = 4
    TARGET_UPDATE_PERIOD = 100

    # Logging
    # ---------
    EVALUATION_FREQUENCY = 500

    parser = argparse.ArgumentParser(description='DDQL Training for Shortest Path problem')
    parser.add_argument('--max_train_steps', type=int, default=1_000_000,
                        help='Maximum number of training steps (default: 1_000_000)')
    parser.add_argument('--plot',  default=False, action='store_true',
                        help='Plot learning curve (default: False)')
    parser.add_argument('--verbose',  default=False, action='store_true',
                        help='Output training logs (default: False)')
    args = parser.parse_args()

    main(args)