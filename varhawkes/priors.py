import numpy as np

import torch


class Prior:

    def __init__(self, dim, n_params, C):
        self.dim = dim
        if not isinstance(C, torch.Tensor):
            raise ValueError('Parameter `C` should be a tensor of length `n_params`.')
        self.n_params = len(C)
        self.C = C

    def logprior(self, z):
        """
        Log Prior
        """
        raise NotImplementedError('Must be implemented in child class')

    def opt_hyper(self, z):
        """
        Optimal regularization weights for the current value of z
        """
        raise NotImplementedError('Must be implemented in child class')


class GaussianPrior(Prior):

    def logprior(self, z):
        """
        Log Prior
        """
        return -torch.sum((z ** 2) / self.C)

    def opt_hyper(self, z):
        """
        Optimal regularization weights for the current value of z
        """
        return 2 * z ** 2


class GaussianLaplacianPrior(Prior):
    """
    Gaussian prior for baseline intensity and Laplacian prior for adjacency
    """

    def logprior(self, z):
        """
        Prior
        """
        return - torch.sum(z[:self.dim] ** 2 / self.C[:self.dim]) \
               - torch.sum(z[self.dim:] / self.C[self.dim:])
    
    def opt_hyper(self, z):
        """
        Optimal regularization weights for the current value of z
        """
        opt_C = torch.zeros_like(self.C)
        opt_C[:self.dim] = 2 * z[:self.dim]**2
        opt_C[self.dim:] = z[self.dim:]
        return opt_C
