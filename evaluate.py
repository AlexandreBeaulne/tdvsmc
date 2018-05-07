
import argparse
from random import randint
from statistics import mean

import torch
from torch.autograd import Variable as Var
import torch.nn.functional as F

from envs import create_atari_env
from model import ActorCritic

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='pytorch serialized model')
    args = parser.parse_args()

    name, _ext = args.model.split('.')
    algo, game, num_steps = name.split('_')

    rewards = []

    for i in range(1):

        env = create_atari_env(game)

        model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
        model.eval()
        model.load_state_dict(torch.load(args.model))
        cx = Var(torch.zeros(1, 256), volatile=True)
        hx = Var(torch.zeros(1, 256), volatile=True)

        state = env.reset()
        state = torch.from_numpy(state)
        rewards.append(0)
        done = False

        while not done:
            cx, hx = Var(cx.data, volatile=True), Var(hx.data, volatile=True)
            _, logit, (hx, cx) = model((Var(state.unsqueeze(0), volatile=True), (hx, cx)))
            prob = F.softmax(logit)
            action = prob.max(1, keepdim=True)[1].data.numpy()
            state, reward, done, _ = env.step(action[0, 0])
            rewards[-1] += reward
            state = torch.from_numpy(state)
            env.render()

    env.close()

    print('algo,game,rollout,avg_score')
    print('{},{},{},{}'.format(algo, game, num_steps, mean(rewards)))



if __name__ == '__main__':
    main()

