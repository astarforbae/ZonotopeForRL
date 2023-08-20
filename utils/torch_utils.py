################################################################################
#                                author: yy                                    #
#                       Reference:Shangtong Zhang DeepRL                       #
#                    https://github.com/ShangtongZhang/DeepRL/                 #
#                                                                              #
################################################################################
import numpy as np

from .config import *
import torch
import os


def select_device(gpu_id):
    if gpu_id >= 0:
        Config.DEVICE = torch.device("cuda:%d" % (gpu_id))
    else:
        Config.DEVICE = torch.device("cpu")


def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.array(x, dtype=np.float32)
    x = torch.from_numpy(x).to(Config.DEVICE)
    return x


def range_tensor(end):
    return torch.arange(end).long().to(Config.DEVICE)


def to_np(t):
    return t.cpu().detach().numpy()


def random_seed(seed=None):
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))

def set_one_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


