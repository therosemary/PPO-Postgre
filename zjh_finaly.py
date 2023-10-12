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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001  # 学习率
num_steps = 300  # 每回合最多行动次数
beach_size = 300  # 每批次数据最多行动次数
mini_batch_size = 50  # 每次迭代算法的行动次数
gamma = 0.01        # 奖励gamma
clipping = 0.2

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
        self.agent = Agent(env_obj).to(device)

    def get_learning_data(self):
        beach_count = 0
        env_tmp = self.envs

        # 每个步长收集的信息
        batch_state = torch.zeros((beach_size, 1) + env_tmp.observation_space.shape).to(device)
        batch_actions = torch.zeros((beach_size, 1) + env_tmp.action_space.shape).to(device)
        batch_values = torch.zeros(beach_size).to(device)
        batch_reward = torch.zeros(beach_size).to(device)
        batch_log_prob = torch.zeros(beach_size).to(device)
        batch_done = torch.zeros(beach_size).to(device)

        state_begin = torch.Tensor(env_tmp.reset()[0]).to(device)

        next_done = torch.zeros(1).to(device)

        # 小批次收集的信息
        batch_episode_lens = []
        for step in range(beach_size):
            done = False
            # state_begin = env_tmp.reset()
            batch_done[step] = next_done

            # print('起始位置:{}'.format(state_begin))
            each_epi = 0
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(state_begin)
                batch_values[step] = value.flatten()
            batch_actions[step] = action
            batch_state[step] = state_begin
            batch_actions[step] = action
            batch_log_prob[step] = logprob

            env_tmp.render()
            state_begin, reward, done = env_tmp.step(action.cpu().numpy())[0:3]
            batch_reward[step] = reward

            done_ = [1] if done else [0]
            state_begin, next_done = torch.Tensor(state_begin).to(device), torch.Tensor(done_).to(device)

            if done:
                state_begin = torch.Tensor(env_tmp.reset()[0]).to(device)

            beach_count += 1
            each_epi = each_epi + 1

            batch_episode_lens.append(each_epi + 1)

        batch_state = torch.tensor(batch_state, dtype=torch.float)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float)
        batch_log_prob = torch.tensor(batch_log_prob, dtype=torch.float)
        batch_wards_to_go, batch_advantage = self.compute_rewards(batch_reward, batch_values, batch_done)
        return batch_state, batch_actions, batch_values, batch_wards_to_go, batch_advantage, batch_log_prob, batch_episode_lens


    def learn(self, target_times):
        import matplotlib.pyplot as plt
        optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-5)
        current_times = 1
        actor_loss_list = []
        critic_loss_list = []
        while current_times < target_times + 1:
            # 拿取一批量的数据
            batch_state, batch_actions, batch_values, batch_wards_to_go, batch_advantage, batch_log_prob, batch_episode_lens = self.get_learning_data()
            print('数据生产完毕：{}'.format(current_times))
            b_inds = np.arange(beach_size)
            clipfracs = []

            for start in range(0, beach_size, mini_batch_size):
                end = start + mini_batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(batch_state[mb_inds], batch_actions[mb_inds])
                logratio = newlogprob - batch_log_prob[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clipping).float().mean().item()]

                mb_advantages = batch_advantage[mb_inds]

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clipping, 1 + clipping)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)

                v_loss = 0.5 * ((newvalue - batch_wards_to_go[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - 0.01 * entropy_loss + v_loss * 0.5



                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                optimizer.step()

                actor_loss_list.append(pg_loss.item())
                critic_loss_list.append(v_loss.item())

            current_times += 1

        a = np.array(actor_loss_list)
        b = np.array(critic_loss_list)
        np.savetxt('./loss.txt', (a, b))


        plt.plot(range(1, target_times * 6 + 1), actor_loss_list, 'b-')
        plt.plot(range(1, target_times * 6 + 1), critic_loss_list, 'r-')
        plt.title('Loss')
        plt.show()

if __name__ == '__main__':
    # envs__ = gym.make('MountainCar-v0', render_mode='human')
    # envs__ = gym.make('LunarLander-v2', render_mode='human')
    envs__ = gym.make('LunarLander-v2')
    handler = PPOHandler(envs__)
    handler.learn(50)
