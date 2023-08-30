# main function
import gym

from core.zonotope import ZonotopeAgent


def train(args):
    env = gym.make(args.env_name)

    env.reset()
    model = ZonotopeAgent(env)
    # 范围

    model.train()
    env.close()
