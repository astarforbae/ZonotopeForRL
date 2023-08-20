################################################################################
#                                author: yy                                    #
#                       Reference:Shangtong Zhang DeepRL                       #
#                    https://github.com/ShangtongZhang/DeepRL/                 #
#                                                                              #
################################################################################


import numpy as np
import pickle
import os
import datetime
import torch
import time

from .torch_utils import *
from pathlib import Path
import itertools
from collections import OrderedDict, Sequence


def run_steps(agent):
    """

    :param agent:
    """
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save('data/%s-%s-%d' % (agent_name, config.tag, agent.total_steps))
        if config.log_interval and not agent.total_steps % config.log_interval:
            agent.logger.info('steps %d, %.2f steps/s' % (agent.total_steps, config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()
        agent.switch_task()


def get_time_str():
    """
    :return: return datetime of now, the format is %y%m%d-%H%M%S
    """
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_default_log_dir(name):
    """
    Get the name of the log dir
    :param name: the name parameter, usually is the agent name
    :return: the dir name
    """
    return './log/%s-%s' % (name, get_time_str())


def mkdir(path):
    """
    Make a new dir according to the path
    :param path: the dir path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def close_obj(obj):
    """
    close the object
    :param obj
    """
    if hasattr(obj, 'close'):
        obj.close()


def random_sample(indices, batch_size):
    """
    sample
    :param indices:
    :param batch_size:
    :return:
    """
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]

def is_plain_type(x):
    """
    Check the type of the x, if it is a plain type return True
    :param x: the object to be checked
    :return: if it is a plain object, return True
    """
    for t in [str, int, float, bool]:
        if isinstance(x, t):
            return True
    return False

