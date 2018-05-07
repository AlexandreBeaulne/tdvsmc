
import argparse
import os

import torch
import torch.multiprocessing as mp

import a3c
from envs import create_atari_env
from envs import games
from model import ActorCritic
import my_optim
import nstepqlearning
from test import test


# Based on https://github.com/pytorch/examples/tree/master/mnist_hogwild

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--game', default='pong', choices=games.keys())
parser.add_argument('--algo', default='nstepQlearning', choices={'nstepQlearning', 'A3C'})
parser.add_argument('--total-steps', type=int, default=60000000,
                    help='total number of steps to train')


if __name__ == '__main__':

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    env = create_atari_env(args.game)

    policy_model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    policy_model.share_memory()

    optimizer = my_optim.SharedAdam(policy_model.parameters(), lr=args.lr)
    optimizer.share_memory()

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    if args.algo == 'nstepQlearning':
        algo = nstepqlearning.train
        target_model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
        target_model.load_state_dict(policy_model.state_dict())
        target_model.share_memory()
        arguments = [args, policy_model, target_model, counter, lock, optimizer]
    elif args.algo == 'A3C':
        algo = a3c.train
        arguments = [args, policy_model, counter, lock, optimizer]

    processes = []

    p = mp.Process(target=test, args=(args.num_processes, args, policy_model, counter))
    p.start()
    processes.append(p)

    for rank in range(args.num_processes):
        p = mp.Process(target=algo, args=arguments + [args.seed + rank])
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

