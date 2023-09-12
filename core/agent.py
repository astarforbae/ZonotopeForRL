################################################################################
#                                author: yy                                    #
#                                agent.py                                      #
#                                                                              #
#                                                                              #
################################################################################
import random

import numpy as np
import torch
from scipy.optimize import linprog
from torch import nn, optim
from torch.nn import functional as func

from .reply_buffer import ReplayBuffer
from .models import Actor, Critic
from .zonotope import is_in_zonotope

from utils.config import Config
from utils.logger import *


class ZonotopeAgent:
    """
    Agent using DDPG training and use zonotope to divide the state space
    """

    def __init__(self,
                 env,
                 num_interval=10,
                 num_episodes=100,
                 max_steps=15000,
                 actor_lr=0.1,
                 critic_lr=0.1,
                 gamma=0.99,
                 tau=0.05,
                 hidden_layer_size=64,
                 e_greedy=0.5,
                 discount_factor=0.99,
                 replay_capacity=10000,
                 batch_size=32,
                 learn_iter=16,
                 save_iter=16):
        # basic argument
        self.env = env
        self.num_interval = num_interval
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.e_greedy = e_greedy
        self.discount_factor = discount_factor
        self.num_episodes = num_episodes
        self.replay_capacity = replay_capacity
        self.batch_size = batch_size
        self.learn_iter = learn_iter
        self.save_iter = save_iter
        # logger
        self.logger = get_logger()
        # actor critic
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.shape[0]
        # the input size is the sum of center vec and generate_matrix
        actor_input_size = self.s_dim * self.s_dim + self.s_dim
        # the output size
        actor_output_size = self.a_dim
        self.actor = Actor(actor_input_size, hidden_layer_size, actor_output_size)
        self.target_actor = Actor(actor_input_size, hidden_layer_size, actor_output_size)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        critic_input_size = actor_input_size + self.s_dim + self.a_dim
        critic_output_size = self.a_dim
        self.critic = Critic(critic_input_size, hidden_layer_size, critic_output_size)
        self.target_critic = Critic(critic_input_size, hidden_layer_size, critic_output_size)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        # zonotope
        self.zonotope_mapping = self.generate_zonotope_mapping(env.observation_space, num_interval)
        # TODO: divide_tools
        self.divide_tools = None
        # TODO: zonotope data structure
        self.kd_tree = None
        # memory buffer
        self.memory = ReplayBuffer(replay_capacity)
        # reward_list
        self.reward_list = []

    def save(self):
        """
        实现模型的保存
        """
        pt_name0 = os.path.join("../pt/actor.pt")
        pt_name1 = os.path.join("../pt/critic.pt")
        pt_name2 = os.path.join("../pt/target_actor.pt")
        pt_name3 = os.path.join("../pt/target_critic.pt")
        torch.save(self.actor.state_dict(), pt_name0)
        torch.save(self.critic.state_dict(), pt_name1)
        torch.save(self.target_actor.state_dict(), pt_name2)
        torch.save(self.target_critic.state_dict(), pt_name3)

    def load(self):
        """
        实现4个模型的加载
        """

        pt_name0 = "../pt/actor.pt"
        pt_name1 = "../pt/critic.pt"
        pt_name2 = "../pt/target_actor.pt"
        pt_name3 = "../pt/target_critic.pt"
        self.actor.load_state_dict(torch.load(pt_name0))
        self.critic.load_state_dict(torch.load(pt_name1))
        self.target_actor.load_state_dict(torch.load(pt_name2))
        self.target_critic.load_state_dict(torch.load(pt_name3))

    def generate_zonotope_mapping(self, observation_space, num_interval):
        """
        TODO(yy): 移植到dividetools
        划分zonotope，并使用
        :param observation_space: 状态空间
        :param num_interval: 划分的粒度
        :return:
        """
        # 创建空的 zonotope_mapping 字典
        zonotope_mapping = {}
        step_sizes = (observation_space.high - observation_space.low) / (2 * num_interval)
        split_points = [np.linspace(observation_space.low[i] + step_sizes[i], observation_space.high[i] - step_sizes[i],
                                    num=num_interval, dtype=np.float64)
                        for i in range(len(observation_space.low))]

        # 生成多维网格点
        grid = np.meshgrid(*split_points, indexing='ij')

        # 遍历所有网格点，生成 zonotope，并添加到字典中
        for i in range(grid[0].size):
            center_vec = np.array([g.ravel()[i] for g in grid])  # Zonotope 的中心向量
            generate_matrix = np.diag(step_sizes)  # 生成矩阵为对角阵，对角线元素为每个区间的一半长度
            zonotope_mapping[i] = (center_vec, generate_matrix)

        return zonotope_mapping

    def is_in_zonotope(self, state, center_vec, generate_matrix):
        """
        TODO:迁移到divide_tools中，结合KDTREE
        判断是否在zonotope中
        :param generate_matrix:
        :param center_vec:
        :param state:
        :return:
        """

        c = np.ones(generate_matrix.shape[1])  # 目标
        A_ub = np.vstack((generate_matrix.T, -generate_matrix.T))  # 不等式上下界限
        b_ub = np.ones(2 * generate_matrix.shape[1])  #
        b_eq = state - center_vec  # Right-hand side of equalities
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=generate_matrix, b_eq=b_eq, method='highs', options={'tol': 1e-5})

        return res.success

    def find_zonotope(self, state, zonotope_mapping):
        """
        TODO:使用空间索引加快搜索进度
        find id of zonotope including the state
        :param state -- numpy tuple
        :param zonotope_mapping --  dict id->zonotope

        :return id if find the zonotope else None。
        """
        for _, (center_vec, generate_matrix) in zonotope_mapping.items():
            # 判断状态是否在zonotope内部
            if is_in_zonotope(state, center_vec, generate_matrix):
                return center_vec, generate_matrix
        print(1)
        return zonotope_mapping[0]

    def update_egreed(self):
        """
        更新egreed
        :return:
        """
        self.e_greedy = max(0.0001, self.e_greedy - 0.001)

    def get_action(self, s0):
        """
        先判断zonotope编号
        神经网络输入zonotope然后得到线性
        :param s0:
        :return:
        """
        zonotope = self.find_zonotope(s0, self.zonotope_mapping)
        c, g = zonotope
        abs_s0 = np.append(c, g.flatten())
        s0 = torch.tensor(abs_s0, dtype=torch.float).unsqueeze(0)
        a0 = self.actor(s0).squeeze(0).detach().numpy()
        return a0

    def step(self):
        """
        每一轮训练训练
        :return:
        """
        # 经验池回放
        if len(self.memory) < self.batch_size:
            return
        s0, a0, r, s1 = self.memory.sample(self.batch_size)

        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        r = torch.tensor(r, dtype=torch.float).view(self.batch_size, -1)
        s1 = torch.tensor(s1, dtype=torch.float)

        def _critic_learn():
            a1 = self.target_actor(s1).detach()
            y_true = r + self.gamma * self.target_critic(s1, a1).detach()

            y_pred = self.critic(s0, a0)

            loss = nn.MSELoss(y_pred, y_true)

            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

        def _actor_learn():
            loss = -torch.mean(self.critic(s0))
            self.actor_optim.zero_grad()
            loss.backward()

        def _soft_update(net_target, net, tau):
            """
            软更新
            :param net_target:
            :param net:
            :param tau:
            :return:
            """
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        _critic_learn()
        _actor_learn()
        _soft_update(self.target_critic, self.critic, self.tau)
        _soft_update(self.target_actor, self.actor, self.tau)

    def eval_model(self):
        """
        TODO 评估一个模型
        :return: 是否失败过
        """
        fail = 0
        min_x1 = 100
        max_x1 = -100
        min_x3 = 100
        max_x3 = -100
        for e in range(100):
            reward = 0
            s0, _ = self.env.reset()
            safe = True
            for step in range(1000):
                a0 = self.get_action(s0)
                s1, r1, done, _ = self.env.step(a0)
                if s1[0] < min_x1:
                    min_x1 = s1[0]
                if s1[0] > max_x1:
                    max_x1 = s1[0]
                if s1[2] < min_x3:
                    min_x3 = s1[2]
                if s1[2] > max_x3:
                    max_x3 = s1[2]
                if done:
                    print(e, ':unsafe', s1, step)
                    fail += 1
                    safe = False
                    break
                reward += r1
                s0 = s1
            if safe:
                print(e, ':safe', s0, min_x1, max_x1, min_x3, max_x3)
            print(fail, '/', 100)
            return fail == 0

    def train(self, terminate_pre=True):
        for episode in range(self.num_episodes):
            s0 = self.env.reset()
            self.update_egreed()
            # self.env.render()
            tot_reward = 0
            step_size = 0
            episode_reward = 0
            for step in range(self.max_steps):
                if np.random.rand() < self.e_greedy:
                    a0 = [(np.random.rand() - 0.5) * 2]  # ?
                else:
                    a0 = self.get_action(s0)
                s1, r, done, _ = self.env.step(a0)
                step_size += 1
                # next_abs =
                # agent.put
                episode_reward += r
                s0 = s1
                # abs = next_abs
                if step % self.learn_iter:  # 更新
                    self.step()

                if done:  # 结束一个轮次
                    break
            if episode % self.save_iter:
                self.save()
            self.reward_list.append(episode_reward)
            # TODO logger print
            if terminate_pre and episode >= 500:
                res = self.eval_model()
                print(res)
                return
