################################################################################
#                                author: yy                                    #
#                                zonotope.py                                   #
#                                                                              #
################################################################################

import numpy as np
import torch
from scipy import optimize as optim


def is_in_zonotope(p, c, G):
    """
       检查点 p 是否在由中心 c 和生成矩阵 G 定义的 zonotope 内。

       参数:
           p (numpy.ndarray): 要检查的点。
           c (numpy.ndarray): Zonotope的中心。
           G (numpy.ndarray): Zonotope的生成矩阵。

       返回:
           bool: 如果点在zonotope内，则为True。
           numpy.ndarray: 如果点在zonotope内，返回该点。
       """
    point = np.array(p)
    center_vec = np.array(c)
    generate_matrix = np.array(G)

    c = np.zeros(generate_matrix.shape[1])  # 不关心的解
    A_eq = generate_matrix.T  # 合并生成矩阵和其负数，用于设置约束
    b_eq = point - center_vec  # 将问题转换为相对于中心点的坐标

    bounds = [(-1, 1) for _ in range(generate_matrix.shape[1])]  # 系数在[0,1]之间

    res = optim.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    return res.success  # 如果找到了一个解，说明点在 Zonotope 内部


def find_zonotope(state, zonotope_mapping):
    """
    TODO:使用空间索引加快搜索进度
    find id of zonotope including the state
    :param state -- numpy tuple
    :param zonotope_mapping --  dict id->zonotope

    :return id if find the zonotope else None。
    """
    for _, (center_vec, generate_matrix) in zonotope_mapping.items():
        # 判断状态是否在zonotope内部
        if is_in_zonotope(state, center_vec, generate_matrix):
            return center_vec, generate_matrix
    return zonotope_mapping[0]


def generate_zonotope_mapping(observation_space, num_interval):
    """
    划分zonotope
    :param observation_space: 状态空间
    :param num_interval: 划分的粒度
    :return:
    返回状态空间的均分网格使得可以判断属于那个网格之中
    """
    # 创建空的 zonotope_mapping 字典
    zonotope_mapping = {}
    step_sizes = (observation_space.high - observation_space.low) / (2 * num_interval)
    split_points = [np.linspace(observation_space.low[i] + step_sizes[i], observation_space.high[i] - step_sizes[i],
                                num=num_interval, dtype=np.float64)
                    for i in range(len(observation_space.low))]

    # 生成多维网格点
    grid = np.meshgrid(*split_points, indexing='ij')

    # 遍历所有网格点，生成 zonotope，并添加到字典中
    for i in range(grid[0].size):
        center_vec = np.array([g.ravel()[i] for g in grid])  # Zonotope 的中心向量
        generate_matrix = np.diag(step_sizes)  # 生成矩阵为对角阵，对角线元素为每个区间的一半长度
        zonotope_mapping[i] = (center_vec, generate_matrix)

    return zonotope_mapping


def to_abstract(state, zonotope_mapping):
    """
    将具体状态变换成输入神经网络的抽象状态
    :param state 具体状态
    :param zonotope_mapping 存储所有zonotope的集合
    :return 抽象状态abs_state
    """
    c, G = find_zonotope(state, zonotope_mapping)
    # 拼接
    flatten_c = np.ravel(c)
    flatten_G = np.ravel(G)
    abs_state = np.concatenate([flatten_c, flatten_G])

    abs_state = torch.from_numpy(abs_state)
    return abs_state.float().unsqueeze(0)
