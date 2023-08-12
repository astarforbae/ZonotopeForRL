import numpy as np
import torch
from scipy.optimize import linprog
from torch import nn, optim
from torch.nn import functional as func

from .reply_buffer import ReplayBuffer
from .agent import AbstractAgent





class ZonotopeAgent(AbstractAgent):
    """
    :keyword
    """
    def __init__(self, env, num_interval=10, num_episodes=100, learning_rate=0.1, discount_factor=0.99,
                 replay_capacity=10000, batch_size=32):
        super(ZonotopeAgent, self).__init__(env)
        self.zonotope_mapping = self.generate_zonotope_mapping(env.observation_space, num_interval)
        self.agent_mapping, self.critic_mapping = self.generate_controller_mapping(self.zonotope_mapping, learning_rate)
        self.num_episodes = num_episodes
        self.step_size = None
        self.discount_factor = discount_factor  # 用于计算未来奖励的折扣因子
        self.memory = ReplayBuffer(replay_capacity)
        self.batch_size = batch_size

    def generate_zonotope_mapping(self, observation_space, num_interval):
        """
        :param observation_space: 状态空间
        :param num_interval: 划分的粒度
        :return:
        """
        # 创建空的 zonotope_mapping 字典
        zonotope_mapping = {}
        step_sizes = (observation_space.high - observation_space.low) / (2 * num_interval)
        split_points = [np.linspace(observation_space.low[i] + step_sizes[i], observation_space.high[i] - step_sizes[i],
                                    num=num_interval)
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

    def generate_controller_mapping(self, zonotope_mapping, hidden_size=128, learning_rate=0.1):
        """
        :param zonotope_mapping: id->zonotope
        :return: linear controller mapping
        """
        state_dim = next(iter(zonotope_mapping.values()))[0].shape[0]
        feature_dim = state_dim + state_dim * state_dim  # 中心向量的大小 + 生成矩阵的大小
        model = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        return model, optimizer

    def find_zonotope(self, state, zonotope_mapping):
        """
        find id of zonotope including the state

        :param state -- numpy tuple
        :param zonotope_mapping --  dict id->zonotope

        :return id if find the zonotope else None。
        """
        for z_id, (center_vec, generate_matrix) in zonotope_mapping.items():
            # 判断状态是否在zonotope内部
            if self.is_in_zonotope(state, center_vec, generate_matrix):
                return z_id

        return None

    def compute_loss(self, batch_size=32):
        """
        计算并返回Critic和Actor的损失
        """
        if len(self.memory) < batch_size:
            return None, None

        transitions = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.uint8)

        # 计算Critic的损失
        q_values = torch.cat(
            [self.critic_mapping[self.find_zonotope(state, self.zonotope_mapping)][0](state) for state in states])
        next_q_values = []
        for next_state in next_states:
            zonotope_id = self.find_zonotope(next_state, self.zonotope_mapping)
            if zonotope_id is None:
                next_q_value = torch.tensor(0.0, dtype=torch.float32)  # 使用默认值
            else:
                next_q_value = self.critic_mapping[zonotope_id][0](next_state)
            next_q_values.append(next_q_value)
        next_q_values = torch.stack(next_q_values)
        expected_q_values = rewards + self.discount_factor * next_q_values * (1 - dones)
        critic_loss = func.mse_loss(q_values, expected_q_values.detach())

        # 计算Actor的损失
        policy_loss = -q_values.mean()

        return policy_loss, critic_loss

    def train(self):
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            done = False
            self.env.render()
            while not done:
                zonotope_id = self.find_zonotope(state, self.zonotope_mapping)
                if zonotope_id is None:
                    print(f"{'-' * 10}")
                    # controller = self.controller_mapping[0]
                    break
                else:
                    center_vec, generate_matrix = self.zonotope_mapping[zonotope_id]
                    input_vec = np.concatenate((center_vec, generate_matrix.flatten()))  # 合并中心向量和生成矩阵
                    controller_params = self.controller_mapping[0](torch.tensor(input_vec, dtype=torch.float32))

                if controller_params is not None:
                    action = controller_params * state  # 使用线性控制器生成动作

                else:
                    action = self.env.action_space().sample()

                next_state, reward, done, _, _ = self.env.step([action])

                # 存储过去的经验
                self.memory.push(state, action, reward, next_state, done)

                # 计算损失并进行优化
                policy_loss, critic_loss = self.compute_loss()
                if policy_loss is not None and critic_loss is not None:
                    agent[1].zero_grad()
                    policy_loss.backward()
                    agent[1].step()

                    critic[1].zero_grad()
                    critic_loss.backward()
                    critic[1].step()

                state = next_state
