# main function
import gym

from core.agent import ZonotopeAgent


def train(args):
    env = gym.make(args.env_name)

    env.reset()
    model = ZonotopeAgent(env, is_abstract=False)
    # 范围

    model.train()
    env.close()
