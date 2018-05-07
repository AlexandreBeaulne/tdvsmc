
from random import random

import torch
from torch.autograd import Variable

from envs import create_atari_env
from model import ActorCritic
from utils import ensure_shared_grads
from utils import Epsilon

def train(args, policy_model, target_model, counter, lock, optimizer, seed):

    torch.manual_seed(seed)

    env = create_atari_env(args.game)
    env.seed(seed)

    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(policy_model.state_dict())
    model.train()

    epsilon = Epsilon(1, 0.1, 1000000)

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    while counter.value < args.total_steps:

        # Sync with the shared model
        model.load_state_dict(policy_model.state_dict())
        if done:
            cx, hx = Variable(torch.zeros(1, 256)), Variable(torch.zeros(1, 256))
            cx_tgt, hx_tgt = Variable(torch.zeros(1, 256)), Variable(torch.zeros(1, 256))
        else:
            cx, hx = Variable(cx.data), Variable(hx.data)
            cx_tgt, hx_tgt = Variable(cx_tgt.data), Variable(hx_tgt.data)

        qvalues, rewards = [], []

        for _ in range(args.num_steps):
            episode_length += 1

            _, logit, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))
            _, _, (hx_tgt, cx_tgt) = target_model((Variable(state.unsqueeze(0)), (hx_tgt, cx_tgt)))

            if random() > epsilon(counter.value):
                action = logit.max(1, keepdim=True)[1].data[0, 0]
            else:
                action = env.action_space.sample()

            qvalue = logit[0, action][0]

            state, reward, done, _ = env.step(action)
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            qvalues.append(qvalue)
            rewards.append(reward)

            with lock:
                counter.value += 1
                if counter.value % 20000 == 0:
                    target_model.load_state_dict(policy_model.state_dict())

            if done:
                episode_length = 0
                state = env.reset()
                state = torch.from_numpy(state)
                break
            else:
                state = torch.from_numpy(state)


        if done:
            R = 0
        else:
            _, logit, _ = target_model((Variable(state.unsqueeze(0)), (hx_tgt, cx_tgt)))
            R = logit.max(1, keepdim=True)[0].data[0, 0]

        loss = 0

        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            loss = loss + (R - qvalues[i]).pow(2)

        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, policy_model)
        optimizer.step()

