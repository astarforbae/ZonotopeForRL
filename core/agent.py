################################################################################
#                                author: yy                                    #
#                                agent.py                                      #
#                                                                              #
################################################################################
import torch
from torch import nn, optim

from utils.logger import *
from utils.misc import get_default_pt_dir
from .models import Actor, Critic
from .reply_buffer import ReplayBuffer
from .zonotope import generate_zonotope_mapping, to_abstract


class ZonotopeAgent:
    """
    Agent using DDPG training and use zonotope to divide the state space
    """

    def __init__(self,
                 env,
                 num_interval=10,
                 num_episodes=500,
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
        # zonotope 划分
        self.zonotope_mapping = generate_zonotope_mapping(env.observation_space, num_interval)
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
        pt_name0 = get_default_pt_dir('actor.pt')
        pt_name1 = get_default_pt_dir("critic.pt")
        pt_name2 = get_default_pt_dir("target_actor.pt")
        pt_name3 = get_default_pt_dir("target_critic.pt")
        torch.save(self.actor.state_dict(), pt_name0)
        torch.save(self.critic.state_dict(), pt_name1)
        torch.save(self.target_actor.state_dict(), pt_name2)
        torch.save(self.target_critic.state_dict(), pt_name3)

    def load(self):
        """
        实现4个模型的加载
        """

        pt_name0 = os.path.join(Config.PT_FOLDER, "actor.pt")
        pt_name1 = os.path.join(Config.PT_FOLDER, "critic.pt")
        pt_name2 = os.path.join(Config.PT_FOLDER, "target_actor.pt")
        pt_name3 = os.path.join(Config.PT_FOLDER, "target_critic.pt")
        self.actor.load_state_dict(torch.load(pt_name0))
        self.critic.load_state_dict(torch.load(pt_name1))
        self.target_actor.load_state_dict(torch.load(pt_name2))
        self.target_critic.load_state_dict(torch.load(pt_name3))

    def update_egreed(self):
        """
        更新egreed
        :return:
        """
        self.e_greedy = max(0.0001, self.e_greedy - 0.001)

    def get_action(self, abs_s0):
        """
        先判断zonotope编号
        神经网络输入zonotope然后得到线性
        :param abs_s0:抽象状态
        :return:
        """
        a0 = self.actor(abs_s0).squeeze(0).detach().numpy()
        return a0

    def step(self):
        """
        每一轮训练训练
        :return:
        """
        # 经验池回放
        if len(self.memory) < self.batch_size:
            return
        """
        此时的memory是abs_s0, a0, r, s1, done 为一组，有很多组
        我们需要的是把每一列作为一个数组
        """
        s0, abs_s0, a0, r, s1, abs_s1, _ = tuple([list(col) for col in zip(*self.memory.sample(self.batch_size))])
        s0 = torch.from_numpy(np.array(s0))
        abs_s0 = torch.stack(abs_s0, dim=0).squeeze(1)
        a0 = torch.tensor(a0, dtype=torch.float)
        r = torch.tensor(r, dtype=torch.float).view(self.batch_size, -1)
        s1 = torch.from_numpy(np.array(s1))
        abs_s1 = torch.stack(abs_s1, dim=0).squeeze(1)
        concat_s0 = torch.cat([abs_s0, s0], dim=1).float()
        concat_s1 = torch.cat([abs_s1, s1], dim=1).float()

        def _critic_learn():
            a1 = self.target_actor(abs_s1).detach()
            y_true = r + self.gamma * self.target_critic(concat_s1, a1).detach()

            y_pred = self.critic(concat_s0, a0)
            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)

            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

        def _actor_learn():
            """
            更新actor
            """
            loss = -torch.mean(self.critic(concat_s0, self.actor(abs_s0)))
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

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
        评估模型是否争取
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
        self.logger.lazy_init_writer()
        for episode in range(self.num_episodes):
            s0 = self.env.reset()
            abs_s0 = to_abstract(s0, self.zonotope_mapping)
            self.update_egreed()
            step_size = 0
            episode_reward = 0
            for step in range(self.max_steps):
                if np.random.rand() < self.e_greedy:
                    a0 = [(np.random.rand() - 0.5) * 2]  # e_greedy 比较小的情况进行一定的探索
                else:
                    a0 = self.get_action(abs_s0)
                s1, r, done, _ = self.env.step(a0)
                step_size += 1
                abs_s1 = to_abstract(s1, self.zonotope_mapping)
                self.memory.push(s0, abs_s0, a0, r, s1, abs_s1, done)
                episode_reward += r
                abs_s0 = abs_s1
                if step % self.learn_iter:  # 更新
                    self.step()

                if done:  # 结束一个轮次
                    break
            if episode % self.save_iter == 0:
                self.save()
            self.reward_list.append(episode_reward)
            self.logger.info(f'Episode: {episode}, Cumulative_Reward: {episode_reward}')
            self.logger.add_scalar('Cumulative_Reward', episode_reward, episode)

            if terminate_pre and episode >= 500:
                res = self.eval_model()
                print(res)
                return
