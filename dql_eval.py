import numpy as np
import torch
import json
import random
import argparse

from env import GraphEnv
from dql import DoubleDQL


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load learning parameters
    with open(args.path + 'learning/params.dat') as pf:
        params = json.load(pf)

    # Set random generator seed
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    env = GraphEnv(
        num_nodes=params['num_nodes'], destination_node=params['destination_node'],
        bi_prob=params['bi_prob'])

    print(f"env.observation_space.size(): {env.observation_space.shape}")
    print(f"env.action_space.n: {env.action_space.n}")

    # The DoubleDQL class object
    dqn = DoubleDQL(train_env=env, hl1_size=params['hl1_size'],
                    hl2_size=params['hl2_size'], device=device)

    dqn.load_main_dqn(model_path=args.path + 'models/' + (
        'latest_policy.pth' if args.latest else 'best_policy.pth'))

    # Evaluate saved best agent
    dqn.final_evaluation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DDQL Evaluation for Shortest Path problem')
    parser.add_argument('--path', type=str, default=None,
                        help='Path to the trined model (default: None)')
    parser.add_argument('--latest',  default=False, action='store_true',
                        help='Evaluate the latest policy (default: False)')
    args = parser.parse_args()

    main(args)