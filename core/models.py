import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Deep Deterministic Policy Gradient (DDPG) - Actor
    :param 
    """

    def __init__(self, input_size, hidden_size, output_size, num_hidden=3):
        super(Actor, self).__init__()
        larger_weight = 1
        self.input_size = input_size
        self.output_size = output_size
        # network arch
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        ]

        for _ in range(num_hidden):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh()
            ])
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, larger_weight)
                layer.bias.data.zero_()

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
        生成线性控制器的系数
        :param s: 抽象状态
        :return:
        """
        x = self.model(s)
        # b = torch.Tensor([1.0, 2.0, 1.0, 10.0, 1.0])
        # x = x * b
        return x


class Critic(nn.Module):
    """
    :param
    :param
    :param
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
