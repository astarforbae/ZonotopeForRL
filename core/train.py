# main function
import gym

from env.cartpole_continuous import CartPoleContinuousEnv

from core.agent import ZonotopeAgent


def train(args):
    if args.env_name == 'CartPoleContinuous':
        env = CartPoleContinuousEnv()
    else:
        env = gym.make(args.env_name)

    env.reset()
    model = ZonotopeAgent(env, is_abstract=False)
    # 范围

    model.train(terminate_pre=False)
    env.close()
