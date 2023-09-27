import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('./zjh_logs/go')
torch.autograd.set_detect_anomaly(True)


class MyEnv:

    def __init__(self):
        self.started = [0, 0]
        self.action_dim = 2

    def reset(self):
        self.__init__()

    def step(self, action):
        new_state = 1
        reward = 2
        return new_state, reward, False


class ZjhNet(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ZjhNet, self).__init__()
        self.layer1 = self.layer_init(nn.Linear(in_dim, 100))
        self.layer2 = self.layer_init(nn.Linear(100, 100))
        self.layer3 = self.layer_init(nn.Linear(100, out_dim))

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def forward(self, data):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float)
        activation1 = F.relu(self.layer1(data))
        activation2 = F.relu(self.layer2(activation1))
        return self.layer3(activation2)


class PPO:

    def __init__(self, env_):
        s_dim = 8
        a_dim = 2
        self.status_dim = s_dim
        self.action_dim = a_dim
        self.lr = 0.001  # learning rate
        self.actor = ZjhNet(s_dim, a_dim)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic = ZjhNet(s_dim, 1)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        # 一次取样最多的步长数
        self.max_batch = 320
        # 单次episode最多的步长数  因此一次beach可能的episode数量在[2, 3200] 包含小数 可能没结束但是步长够了
        self.max_per_episode = 120
        # 学习效率
        self.gamma = 0.1
        self.env = env_
        self.action_dim = a_dim
        # 通过假设一个标准差 构造一个对角矩阵
        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # 规定每次数据的网络参数迭代次数
        self.max_times_of_net = 6
        # 每次迭代网络参数时允许变更的概率比例
        self.changing_rate = 0.25

    def get_actions(self, state_):
        # 根据状态 用actor网络采用随机梯度方法 给出所选动作以及概率
        # 通过actor网络输出action的平均期望
        # print('位置开始:{}'.format(state_))
        if not isinstance(state_, torch.Tensor):
            state = torch.tensor(state_, dtype=torch.float)
        else:
            state = state_
        action_mean = self.actor(state)
        # print('本次动作:{}'.format(action_mean))
        # 使用均值与假设的标准差 做协方差矩阵
        dist = MultivariateNormal(action_mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # print('本次动作的期望:{}'.format(log_prob.detach()))
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

    def evaluate_v(self, states, actions):
        v_list = self.critic(states).squeeze()
        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(states)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(actions)
        return v_list, log_probs

    def collect_batch_data(self):
        beach_count = 0
        env_tmp = self.env

        # 每个步长收集的信息
        batch_state = []
        batch_actions = []
        batch_reward = []
        batch_log_prob = []
        # 小批次收集的信息
        batch_episode_lens = []
        # while beach_count < self.max_batch:
        for s in range(3):
            done = False
            state_begin = env_tmp.reset()[0]
            # print('起始位置:{}'.format(state_begin))
            each_epi = 0
            tem_reward = []
            while not done:
                action, prob = self.get_actions(state_begin)
                env_tmp.render()
                state_begin, reward, done = env_tmp.step(action)[0:3]
                beach_count += 1
                # print('本次动作奖励：{}, 新的位置:{},状态:{}'.format(reward, state_begin, done))
                # print('-------------------')
                batch_state.append(state_begin)
                batch_actions.append(action)
                batch_log_prob.append(prob)
                tem_reward.append(reward)

                if done:
                    break
                each_epi = each_epi + 1
            print(each_epi)
            batch_episode_lens.append(each_epi + 1)
            batch_reward.append(tem_reward)
        batch_state = torch.tensor(batch_state, dtype=torch.float)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float)
        batch_log_prob = torch.tensor(batch_log_prob, dtype=torch.float)
        batch_rewards_to_go = self.compute_rewards(batch_reward)
        return batch_state, batch_actions, batch_rewards_to_go, batch_log_prob, batch_episode_lens

    def learn(self, target_times):
        current_times = 1
        actor_loss_list = []
        critic_loss_list = []
        while current_times < target_times:
            # 拿取一批量的数据
            batch_state, batch_actions, batch_rewards_to_go, batch_log_prob, batch_episode_lens = self.collect_batch_data()
            print(len(batch_state))

            # 计算优势函数里的V值
            v_, _ = self.evaluate_v(batch_state, batch_actions)

            # 计算优势函数
            a_k = batch_rewards_to_go - v_

            # for this_time in range(self.max_times_of_net):
            for this_time in range(self.max_times_of_net):
                print('-------------------------')
                # 运用此时的网络得出此时的V值以及动作分布概率
                v_, pros = self.evaluate_v(batch_state, batch_actions)
                # 取loss
                changing_rate = torch.exp(pros - batch_log_prob)
                surr1 = a_k * changing_rate
                #   将张量控制在这个区间内 超过取1 + self.changing_rate  小于取1 - self.changing_rate
                surr2 = a_k * torch.clamp(changing_rate, 1 - self.changing_rate, 1 + self.changing_rate)

                actor_loss = (-torch.min(surr1, surr2)).mean()
                new_actor = self.actor
                actor_optima = Adam(new_actor.parameters(), lr=self.lr)
                actor_optima.zero_grad()
                actor_loss.backward(retain_graph=True)
                actor_optima.step()
                self.actor = new_actor

                actor_loss_list.append(actor_loss)

                critic_loss = nn.MSELoss()(v_, batch_rewards_to_go)
                critic_loss_list.append(critic_loss)

                new_critic = self.critic
                critic_optima = Adam(new_critic.parameters(), lr=self.lr)
                critic_optima.zero_grad()
                critic_loss.backward()
                critic_optima.step()
                this_time += 1

            current_times += 1
        plt.plot((1, 6), actor_loss_list, 'b-')
        plt.plot((1, 6), critic_loss_list, 'r-')
        plt.title('return')
        plt.show()


if __name__ == '__main__':
    # def print_env(env_):
    #     print(env_.action_space)
    #     print(type(env_.action_space))
    #     print(env_.observation_space)
    #     print(type(env_.observation_space))
    #     print('*************************************')

    import gym
    env = gym.make('LunarLander-v2', render_mode="human", continuous=True)
    # print_env(env)
    # print(env.action_space)
    # print(env.action_space.shape)
    # print(env.action_space.shape[0])
    # print(type(env.action_space))
    # print(env.observation_space)
    # print(env.observation_space.shape)
    # print(env.observation_space.shape[0])
    # print(type(env.observation_space))
    # for i in range(7):
    #     state_begin = env.reset()[0]
    #     print(state_begin)
    model = PPO(env)
    model.learn(10)
