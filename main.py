import argparse

import torch

from core.train import *
from utils.config import Config
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # 环境列表
    env_list = ["Pendulum-v0", "CartPoleContinuous"]
    # 环境
    model_list = ["triangle", "zonotope"]

    parser = argparse.ArgumentParser(description="linear-control")
    parser.add_argument('--model', type=str, default="zonotope", help='type of model', choices=model_list)
    parser.add_argument('--env_name', type=str, default='Pendulum-v0', help='name of donkey sim environment',
                        choices=env_list)
    parser.add_argument('--render_mode', type=str, default='human', help='render method', choices=['human'])

    args = parser.parse_args()
    train(args)
