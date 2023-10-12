import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class TaskInfo:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = 0.001  # 学习率
        self.total_timesteps = 30000
        self.num_steps = 500  # 每回合最多行动次数
        self.batch_size = 500  # 每批次数据最多行动次数
        self.minibatch_size = 100  # 每批次数据最多行动次数
        self.gamma = 0.01        # 奖励gamma
        self.clipping = 0.2
        self.num_envs = 1        # 并行任务数
        self.anneal_lr = 0
        self.update_epochs = 4
        self.clip_coef = 0.2
        self.max_grad_norm = 0.5
        self.ent_coef = 0.01
        self.vf_coef = 0.5


args = TaskInfo()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            # prod()乘积
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        # https://blog.csdn.net/qq_37388085/article/details/127251550  categorical这个类对
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class PPOHandler:

    def __init__(self, env_obj):
        self.envs = env_obj
        self.agent = Agent(env_obj).to(args.device)


    def learn(self):
        import matplotlib.pyplot as plt
        actor_loss_list = []
        critic_loss_list = []
        optimizer = optim.Adam(self.agent.parameters(), lr=args.learning_rate, eps=1e-5)

        envs = self.envs

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(args.device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(args.device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(args.device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        next_obs = torch.Tensor(envs.reset()[0]).to(args.device)
        next_done = torch.zeros(args.num_envs).to(args.device)
        num_updates = args.total_timesteps // args.batch_size

        for update in range(1, num_updates + 1):
            print('pull data: {}'.format(update))
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, info = envs.step(action.cpu().numpy())[0:4]
                rewards[step] = torch.tensor(reward).to(args.device).view(-1)
                done_ = [1] if done else [0]
                next_obs, next_done = torch.Tensor(next_obs).to(args.device), torch.Tensor(done_).to(args.device)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                returns = torch.zeros_like(rewards).to(args.device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds],
                                                                                  b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)

                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                    actor_loss_list.append(pg_loss.item())
                    critic_loss_list.append(v_loss.item())


        a = np.array(actor_loss_list)
        b = np.array(critic_loss_list)
        np.savetxt('./loss.txt', (a, b))

        times = int((args.batch_size / args.minibatch_size) * args.update_epochs * args.total_timesteps / args.batch_size)

        plt.plot(range(1, times + 1), actor_loss_list, 'b-')
        # plt.plot(range(1, times + 1), critic_loss_list, 'r-')
        plt.title('Loss')
        plt.show()


def get_env_info(envs):
    start = envs.reset()[0]
    print(start)
    next_state, reward, done = envs.step(1)[0:3]
    print(next_state, reward, done)
    next_state, reward, done = envs.step(2)[0:3]
    print(next_state, reward, done)
    next_state, reward, done = envs.step(3)[0:3]
    print(next_state, reward, done)

if __name__ == '__main__':
    # envs__ = gym.make('MountainCar-v0', render_mode='human')
    # envs__ = gym.make('LunarLander-v2', render_mode='human')
    envs__ = gym.make('LunarLander-v2')
    # handler = PPOHandler(envs__)
    # handler.learn()
    get_env_info(envs__)