import torch
import numpy as np
import math


class Excitation:

    def __init__(self, cut_off=float("inf")):
        """
        cut_off : float
            Time window size that we calculate the excitation functions for.
        """
        self.cut_off = cut_off

    def call(self, t):
        """
        value of excitation function
        """
        raise NotImplementedError('Must be implemented in child class')

    def callIntegral(self, t):
        """
        Integral of excitation function
        """
        raise NotImplementedError('Must be implemented in child class')


class ExponentialKernel(Excitation):

    def __init__(self, decay, cut_off):
        """
        Exponential kernel
        k(t) = decay x exp(-decay * t)
        K(t) = int_0^t k(s) ds = 1- exp(-decay * t)

        Arguments:
        ----------
        decay : float
            decaying rate
        M : int
            The number of basis functions
        """
        super(ExponentialKernel, self).__init__(cut_off)
        self.decay = decay
        self.M = 1

    def call(self, t):
        """
        value of excitation function
        """
        return self.decay * torch.exp(- self.decay * t)

    def callIntegral(self, t):
        """
        Integral of excitation function
        """
        return 1 - torch.exp(- self.decay * t)


class GaussianFilter(Excitation):

    def __init__(self, t_m, sigma, cut_off):
        """
        Gaussian Filter kernel
        k(t, t_m) = (2 pi sigma^2)^-1 x exp(-(t-t_m)^2 / (2 sigma^2))
        K(t, t_m) = int_0^t k(s, t_m) ds = (2 pi sigma^2)^-1 * sqrt(pi /2) * sigma * [erf{t_m/(sigma sqrt(2))} + erf{(t-t_m)/(sigma sqrt(2))} ]

        Arguments:
        ----------
        t_m : torch.float
        sigma:  torch.float
        M : int
            The number of basis functions
        """
        super(GaussianFilter, self).__init__(cut_off)
        self.t_m = t_m
        self.sigma = sigma
        self.cons1 = math.sqrt(math.pi/2) * torch.erf(self.t_m / (sigma * math.sqrt(2))) / (2 * math.pi * sigma)
        self.M = 1

    def call(self, t):
        """
        value of excitation function
        """
        if len(t) == 0:
            return torch.tensor([0], dtype=torch.float64)
        else:
            return torch.exp(-(t-self.t_m)**2 / (2 * self.sigma**2)) / (2 * math.pi * self.sigma**2)

    def callIntegral(self, t):
        """
        Integral of excitation function
        """
        if len(t) == 0:
            return torch.tensor([0], dtype=torch.float64)
        else:
            return self.cons1 + math.sqrt(math.pi/2) * torch.erf((t-self.t_m) / (self.sigma * math.sqrt(2))) / (2 * math.pi * self.sigma)


class MixtureGaussianFilter(Excitation):

    def __init__(self, M, end_time, cut_off):
        """
        Mixture of Gaussian Filter kernels
        k(t, t_m) = (2 pi sigma^2)^-1 x exp(-(t-t_m)^2 / (2 sigma^2))
        K(t, t_m) = int_0^t k(s, t_m) ds = (2 pi sigma^2)^-1 * sqrt(pi /2) * sigma * [erf{t_m/(sigma sqrt(2))} + erf{(t-t_m)/(sigma sqrt(2))} ]

        Arguments:
        ----------
        t_m : A vector of torch.float (M x 1)
        sigma:  torch.float
        M : int
            The number of basis functions
        sigma = 1 / w_0
        """
        super(MixtureGaussianFilter, self).__init__(cut_off)
        self.M = M
        self.t_m = torch.arange(0, self.M, dtype=torch.float64) * end_time / self.M
        self.sigma = end_time / (M * math.pi)
        self.GaussianFs = [GaussianFilter(t, self.sigma, cut_off) for t in self.t_m]

    def call(self, t):
        """
        value of excitation function
        """
        return torch.stack([GaussianF.call(t) for GaussianF in self.GaussianFs])

    def callIntegral(self, t):
        """
        Integral of excitation function
        """
        return torch.stack([GaussianF.callIntegral(t) for GaussianF in self.GaussianFs])
