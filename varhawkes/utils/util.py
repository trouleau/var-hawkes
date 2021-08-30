"""
Utility helper functions
"""
import numpy as np
import torch


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = torch.exp(x - x.max())
    return e_x / e_x.sum(dim=0)
