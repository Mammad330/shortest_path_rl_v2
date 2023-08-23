import numpy as np
import os
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


# Saves the parameters above in separate file for reference
def save_learning_params(max_train_steps: int) -> Mapping[str, bool | int | float]:
    learning_params = dict()

    # Environment
    # -----------
    learning_params['seed'] = SEED
    learning_params['num_nodes'] = NUM_NODES
    learning_params['destination_node'] = DESTINATION_NODE
    learning_params['bi_prob'] = BI_PROB

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
         str(datetime.now().hour).zfill(2) + '-' + str(datetime.now().minute).zfill(2) +
         '/')

    # Delete if a directory with the same name already exists
    if os.path.exists(train_path):
        rmtree(train_path)

    # Create empty directories for saving model, parameter and log files
    os.makedirs(train_path)
    os.makedirs(train_path + 'plots')
    os.makedirs(train_path + 'learning')
    os.makedirs(train_path + 'models')

    env = GraphEnv(
        num_nodes=NUM_NODES, destination_node=DESTINATION_NODE, bi_prob=BI_PROB)

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
    dqn = DoubleDQL(train_env=env, loss_fn=mse_loss, gamma=GAMMA, lr=LR, epsilon=EPSILON,
                    replay_mem_size=REPLAY_MEM_SIZE, hl1_size=HL1_SIZE, hl2_size=HL2_SIZE,
                    device=device)

    # Train the agent
    dqn.train(
        training_steps=args.max_train_steps, init_training_period=INITIAL_PERIOD,
        main_update_period=MAIN_UPDATE_PERIOD, target_update_period=TARGET_UPDATE_PERIOD,
        batch_size=BATCH_SIZE, evaluation_freq=EVALUATION_FREQUENCY, show_plot=args.plot,
        path=train_path)


if __name__ == '__main__':
    # Environment
    # -----------
    SEED = 0
    NUM_NODES = 25
    DESTINATION_NODE = 20
    BI_PROB = 0.3

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
    BATCH_SIZE = 32

    # DQL
    # -----
    REPLAY_MEM_SIZE = 100_000
    INITIAL_PERIOD = 5000
    MAIN_UPDATE_PERIOD = 4
    TARGET_UPDATE_PERIOD = 100
    EPISODE_LENGTH = 200

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