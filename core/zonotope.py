################################################################################
#                                author: yy                                    #
#                                zonotope.py                                   #
#                                                                              #
################################################################################

import numpy as np
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
