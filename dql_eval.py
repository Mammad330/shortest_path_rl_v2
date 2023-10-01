import numpy as np
import torch
import json
import random
import argparse
import os

from env import GraphEnv
from dql import DoubleDQL


def main(args):
    # Set device to GPU or CPU based on availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load learning parameters
    with open(args.path + 'learning/params.dat') as pf:
        params = json.load(pf)

    # Set random generator seed
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    # Create the environment
    env = GraphEnv(
        num_nodes=params['num_nodes'], edge_prob=params['edge_prob'],
        destination_node=params['destination_node'], lambda_=params['lambda'],
        trans_prob_low=params['trans_prob_low'],
        trans_prob_high=params['trans_prob_high'],
        graph_path=(args.path if os.path.isfile(args.path + 'data/graph.npy')
                    else None),
        path=(None if os.path.isfile(args.path + 'data/graph.npy') else
              args.path))

    print(f"env.observation_space.size(): {env.observation_space.shape}")
    print(f"env.action_space.n: {env.action_space.n}")

    # Create the DDQL agent
    dqn = DoubleDQL(train_env=env, hl1_size=params['hl1_size'],
                    hl2_size=params['hl2_size'], device=device)

    # Load the saved-best policy model from file
    dqn.load_main_dqn(model_path=args.path + 'models/best_policy.pth')

    # Evaluate the model
    dqn.final_evaluation(eval_mode=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DDQL Evaluation for Shortest Path problem')
    parser.add_argument('--path', type=str, default=None,
                        help='Path to the trained model (default: None)')
    args = parser.parse_args()

    main(args)