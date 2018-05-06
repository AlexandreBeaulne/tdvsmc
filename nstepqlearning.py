
from random import random

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from envs import create_atari_env
from model import ActorCritic

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

class Epsilon(object):

    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def __call__(self, t):
        return self.end + max(0, 1 - t / self.decay) * (self.start - self.end)

def train(args, policy_model, target_model, counter, lock, optimizer, rank):

    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

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
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        states, rewards, actions = [], [], []

        for step in range(args.num_steps):
            episode_length += 1

            if random() > epsilon(counter.value):
                _, logit, (hx, cx) = model((Variable(state.unsqueeze(0), volatile=True), (hx, cx)))
                prob = F.softmax(logit)
                action = prob.max(1, keepdim=True)[1].data.numpy()[0, 0]
            else:
                action = env.action_space.sample()

            state, reward, done, _ = env.step(action)
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            with lock:
                counter.value += 1
                if counter.value % 20000 == 0:
                    target_model.load_state_dict(policy_model.state_dict())

            if done:
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state)
            states.append(state)
            rewards.append(reward)
            actions.append(action)

            if done:
                break

        R = Variable(torch.zeros(1, 1))
        if not done:
            _, logit, _ = target_model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = logit.max(dim=1)[0]

        loss = 0

        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            _, logit, _ = model((Variable(states[i].unsqueeze(0)), (hx, cx)))
            Q = logit[0][actions[i]]
            loss = loss + (R - Q).pow(2)
            #loss = loss + F.smooth_l1_loss(R, Q)

        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, policy_model)
        optimizer.step()

