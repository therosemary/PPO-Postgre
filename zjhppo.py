import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal

status_dim = action_dim = 10

class ZjhNet(nn.Module):

    def __int__(self, in_dim, out_dim):
        super(ZjhNet, self).__int__()
        self.layer1 = nn.Linear(in_dim, 100)
        self.layer2 = nn.Linear(100, 200)
        self.layer3 = nn.Linear(200, out_dim)

    def forward(self, data):
        activation1 = F.relu(self.layer1(data))
        activation2 = F.relu(self.layer2(activation1))
        return self.layer3(activation2)


class PPO:

    def __int__(self, s_dim, a_dim):
        self.status_dim = s_dim
        self.action_dim = a_dim
        self.actor = ZjhNet(s_dim, a_dim)
        self.critic = ZjhNet(s_dim, 1)
        # 一次取样最多的步长数
        self.max_batch = 3200
        # 单次episode最多的步长数  因此一次beach可能的episode数量在[2, 3200] 包含小数 可能没结束但是步长够了
        self.max_per_episode = 2400
        # 学习效率
        self.gamma = 0.1

    def get_actions(self, state):
        # 根据状态 用actor网络采用随机梯度方法 给出所选动作以及概率
        # 通过actor网络输出action的平均期望
        action_mean = self.actor(state)
        # 通过假设一个标准差 构造一个对角矩阵
        cov = torch.diag(torch.full(size=(self.action_dim, ), full_value=0.5))
        # 使用均值与假设的标准差 做协方差矩阵
        dist = MultivariateNormal(action_mean, cov)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

    def compute_rewards(self, batch_rewards):
        res = []
        # 把一个episode只有一个rewards的问题 反向解析出每一个步长的rewards
        for each_list in batch_rewards:
            reward = 0
            for each_reward in reversed(each_list):
                reward = each_reward + reward * self.gamma
                res.append(reward)
        return torch.tensor(res, dtype=torch.float)

    def collect_batch_data(self):
        beach_count = 0
        env = 'gym-env'

        # 每个步长收集的信息
        batch_state = []
        batch_actions = []
        batch_reward = []
        batch_log_prob = []
        # 小批次收集的信息
        batch_episode_lens = []

        while beach_count < self.max_batch:
            state_begin = env.reset()
            each_epi = 0
            tem_reward = []
            for each_epi in range(self.max_per_episode):
                action, prob = self.get_actions(state_begin)
                new_state, reward, done = env.step(action)
                batch_state.append(new_state)
                batch_actions.append(action)
                batch_log_prob.append(prob)
                tem_reward.append(reward)

                if done:
                    break

            batch_episode_lens.append(each_epi+1)
            batch_reward.append(tem_reward)
        batch_state = torch.tensor(batch_state, dtype=torch.float)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float)
        batch_log_prob = torch.tensor(batch_log_prob, dtype=torch.float)
        batch_rewards_to_go = self.compute_rewards(batch_reward)
        return batch_state, batch_actions, batch_rewards_to_go, batch_log_prob, batch_episode_lens
