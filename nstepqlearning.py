
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from envs import create_atari_env
from model import ActorCritic

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, args, shared_model, counter, lock, optimizer):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model1 = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    model1.load_state_dict(shared_model.state_dict())
    model1.train()

    model2 = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    model2.load_state_dict(shared_model.state_dict())
    model2.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    while counter.value < args.total_steps:
        # Sync with the shared model
        model1.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        states, rewards, actions = [], [], []

        for step in range(args.num_steps):
            episode_length += 1

            _, logit, (hx, cx) = model2((Variable(state.unsqueeze(0)), (hx, cx)))
            prob = F.softmax(logit, dim=1)

            action = prob.multinomial().data

            state, reward, done, _ = env.step(action.numpy())
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state)
            states.append(state)
            rewards.append(reward)
            actions.append(action[0][0])

            if done:
                break

        R = Variable(torch.zeros(1, 1))
        if not done:
            _, logit, _ = model1((Variable(state.unsqueeze(0)), (hx, cx)))
            R = logit.max(dim=1)[0]

        loss = 0

        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            _, logit, _ = model2((Variable(states[i].unsqueeze(0)), (hx, cx)))
            Q = logit[0][actions[i]]
            loss += (R - Q).pow(2)
            #loss += F.smooth_l1_loss(R, Q)

        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm(model1.parameters(), args.max_grad_norm)

        ensure_shared_grads(model1, shared_model)
        optimizer.step()

        if counter.value % 20000 == 0:
            model2.load_state_dict(shared_model.state_dict())

