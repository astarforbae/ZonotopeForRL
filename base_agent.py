import torch
import numpy as np

class BaseAgent(object):
    """

    """
    def __init__(self, config):
        self.config = config
        self.logger = get_logger