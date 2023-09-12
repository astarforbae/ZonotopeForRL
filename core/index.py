################################################################################
#                                author: yy                                    #
#                                                                              #
################################################################################
from scipy.spatial import cKDTree
class KDTreeManager:
    """
    KDTree 建立索引
    """
    def __init__(self, center_vec):
        self.ckd_tree = cKDTree()

class DivideTools:
    """
    划分空间并且建立索引
    """
    pass
