
import time
from collections import deque

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from envs import create_atari_env
from model import ActorCritic

def test(rank, args, shared_model, counter):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.game)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while counter.value < args.total_steps:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 256), volatile=True)
            hx = Variable(torch.zeros(1, 256), volatile=True)
        else:
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)

        _value, logit, (hx, cx) = model((Variable(state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(logit)
        action = prob.max(1, keepdim=True)[1].data.numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from getting stuck
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            timestr = time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - start_time))
            msg = ('{}: game {}, algo {}, num steps {}, FPS {:.0f}, '
                   'episode reward {}, episode length {}')
            print(msg.format(timestr, args.game, args.algo,
                             counter.value, counter.value / (time.time() - start_time),
                             reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()

            filename = '{}_{}_{}.pt'.format(args.algo, args.game, args.num_steps)
            torch.save(shared_model.state_dict(), filename)

            time.sleep(60)

        state = torch.from_numpy(state)

