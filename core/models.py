import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
    """
    Deep Deterministic Policy Gradient (DDPG) - Actor
    :param 
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        larger_weight = 0.01
        self.input_size = input_size
        self.output_size = output_size
        as_dim = int(input_size / 2)
        # network arch
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear1.weight.data.normal_(0, larger_weight)
        self.linear1.bias.data.zero_()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2.weight.data.normal_(0, larger_weight)
        self.linear2.bias.data.zero_()
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear3.weight.data.normal_(0, larger_weight)
        self.linear3.bias.data.zero_()
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.linear4.weight.data.normal_(0, larger_weight)
        self.linear4.bias.data.zero_()

    def forward(self, x):
        action, _ = self.fw_imp(x)
        return action

    def fw_imp(self, s):
        """

        :param s:
        :return:
        """
        x = self.cal_coefficients(s[:, 0:self.input_size])

        tmp = s[:, self.input_size:]
        ones = torch.ones((tmp.size(0), 1))
        cat = torch.cat([ones, tmp], dim=-1)
        y = cat.mul(x)
        res = torch.sum(y, dim=1, keepdim=True)
        return res, x

    def cal_coefficients(self, s):
        """
        生成神经网络的系数
        :param s:
        :return:
        """
        x = torch.relu(self.linear1(s))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        # ?
        b = torch.Tensor([1.0, 2.0, 1.0, 10.0, 1.0])
        x = x * b
        return x


class Critic(nn.Module):
    """

    """

    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


