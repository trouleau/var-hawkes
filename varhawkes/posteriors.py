import math
import numpy as np
from scipy.stats import norm, lognorm, truncnorm

import torch


class Posterior:

    def sample_epsilon(self, size):
        """
        Sample an array of epsilons from the normal distribution, with size
        (n_samples, n_weights, n_params)
        """
        raise NotImplementedError('Must be implemented in child class')

    def g(self, eps, alpha, beta):
        """
        Reparamaterization of the approximate log-normal posterior function
        """
        raise NotImplementedError('Must be implemented in child class')

    def logpdf(self, eps, alpha, beta):
        """
        Log Posterior
        """
        raise NotImplementedError('Must be implemented in child class')


class LogNormalPosterior(Posterior):

    def __init__(self, device = 'cpu'):
        self.norm = torch.distributions.Normal(loc=0.0, scale=1.0)
        self.device = 'cuda' if torch.cuda.is_available() and device=='cuda' else 'cpu'

    def sample_epsilon(self, size):
        """
        Sample an array of epsilons from the normal distribution, with size
        (n_samples, n_weights, n_params)
        """
        eps_err = torch.randn(size, dtype=torch.float64, device = self.device, requires_grad=False)
        return eps_err

    def g(self, eps, alpha, beta):
        """
        Reparamaterization of the approximate log-normal posterior function
        """
        return torch.exp(alpha + eps * beta.exp())

    def logpdf(self, eps, alpha, beta):
        z = self.g(eps, alpha, beta)
        sigma = beta.exp()
        log_phi = self.norm.log_prob((z.log() - alpha) / sigma)
        return torch.sum(log_phi - sigma.log() - z.log())

    def mode(self, alpha, beta):
        return torch.exp(alpha - beta.exp() ** 2)

    def mean(self, alpha, beta):
        return torch.exp(alpha + 0.5 * beta.exp() ** 2)

    def median(self, alpha, beta):
        return alpha.exp()

    def variance(self, alpha, beta):
        sigma2 = beta.exp() ** 2
        return (sigma2.exp() - 1) * torch.exp(2 * alpha + sigma2)

    def pdf(self, x, alpha, beta):
        sigma2 = beta.exp() ** 2
        return torch.exp(-((torch.log(x) - alpha) ** 2) / (2 * sigma2)) / (x * torch.sqrt(sigma2 * 2 * np.pi))


class TruncatedNormal(Posterior):

    def __init__(self, bounds):
        self.bounds = torch.tensor(bounds, dtype=torch.float64, requires_grad=False)
        self.norm = torch.distributions.Normal(loc=0.0, scale=1.0)
        self.val = None

    def sample_epsilon(self, size):
        eps_arr = np.random.uniform(low=0.0, high=1.0, size=size)
        return torch.tensor(eps_arr, dtype=torch.float64, requires_grad=False)

    def bounds_rescaled(self, alpha, beta, sigma=None):
        if sigma is None:
            sigma = beta.exp()
        return ((self.bounds - alpha.unsqueeze(1)) / sigma.unsqueeze(1))

    def g(self, eps, alpha, beta, return_bounds_cdf_diff=False):
        sigma = beta.exp()
        bounds_rescaled = self.bounds_rescaled(alpha, beta, sigma)
        bounds_cdf = self.norm.cdf(bounds_rescaled)
        bound_cdf_diff = torch.clamp(bounds_cdf[:, 1] - bounds_cdf[:, 0], 1e-5, 1e5)
        z = alpha + sigma * self.norm.icdf(
            torch.clamp(bounds_cdf[:, 0] + eps * bound_cdf_diff, 1e-5, 1.0-1e-5)
        )
        z = torch.clamp(z, self.bounds[0], self.bounds[1])
        if return_bounds_cdf_diff:
            return z, bound_cdf_diff
        return z

    def logpdf(self, eps, alpha, beta):
        z, bound_cdf_diff = self.g(eps, alpha, beta, True)
        log_num = self.norm.log_prob((z - alpha) / beta.exp())
        log_denom = beta + bound_cdf_diff.log()
        logpdf = log_num - log_denom
        return logpdf.sum()

    def pdf(self, x, alpha, beta):
        bounds_rescaled = self.bounds_rescaled(alpha, beta)
        return truncnorm.pdf(a=bounds_rescaled[:, 0].detach().numpy(),
                             b=bounds_rescaled[:, 1].detach().numpy(),
                             loc=alpha.detach().numpy(), scale=beta.exp().detach().numpy(),
                             x=x.numpy())

    def mode(self, alpha, beta):
        return torch.clamp(alpha.detach(), self.bounds[0], self.bounds[1]).numpy()

    def mean(self, alpha, beta):
        bounds_rescaled = self.bounds_rescaled(alpha, beta)
        return truncnorm.mean(a=bounds_rescaled[:, 0].detach().numpy(),
                              b=bounds_rescaled[:, 1].detach().numpy(),
                              loc=alpha.detach().numpy(), scale=beta.exp().detach().numpy())

    def median(self, alpha, beta):
        bounds_rescaled = self.bounds_rescaled(alpha, beta)
        return truncnorm.median(a=bounds_rescaled[:, 0].detach().numpy(),
                                b=bounds_rescaled[:, 1].detach().numpy(),
                                loc=alpha.detach().numpy(), scale=beta.exp().detach().numpy())

    def variance(self, alpha, beta):
        bounds_rescaled = self.bounds_rescaled(alpha, beta)
        return truncnorm.var(a=bounds_rescaled[:, 0].detach().numpy(),
                             b=bounds_rescaled[:, 1].detach().numpy(),
                             loc=alpha.detach().numpy(), scale=beta.exp().detach().numpy())
