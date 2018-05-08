
import argparse
from random import randint

import gym
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable as Var
import torch.nn.functional as F

from envs import create_atari_env
from model import ActorCritic

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='pytorch serialized model')
    parser.add_argument('--num-frames', type=int, default=1000)
    args = parser.parse_args()

    name, _ext = args.model.split('.')
    algo, game, num_steps = name.split('_')

    env = create_atari_env(game)
    env.seed(3434)
    env2 = gym.make('SeaquestDeterministic-v4')
    env2.seed(3434)
    env2.reset()

    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    model.eval()
    model.load_state_dict(torch.load(args.model))
    cx = Var(torch.zeros(1, 256), volatile=True)
    hx = Var(torch.zeros(1, 256), volatile=True)

    state = env.reset()
    state = torch.from_numpy(state)

    for i in range(args.num_frames):
        cx, hx = Var(cx.data, volatile=True), Var(hx.data, volatile=True)
        _, logit, (hx, cx) = model((Var(state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(logit)
        action = prob.max(1, keepdim=True)[1].data.numpy()
        state, _reward, _done, _ = env.step(action[0, 0])
        img, _, _, _ = env2.step(action[0, 0])
        plt.imsave('images/seaquest_a3c_{}'.format(i), img)
        state = torch.from_numpy(state)
        env2.render()

    env.close()
    env2.close()

if __name__ == '__main__':
    main()

