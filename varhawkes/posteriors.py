"""
Module that implement posterior distributions
"""
import math
import numpy as np
from scipy.stats import norm, lognorm, truncnorm
import torch


class Posterior:
    """
    Implements abstract posterior distributions
    """

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
    """
    Log-Normal posterior distribution
    """

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

        Arguments:
        ----------
        eps : torch.Tensor
            Noise sample
        alpha : torch.Tensor
            Mean parameters
        beta : torch.Tensor
            Standard deviation parameters (in log-scale)
        """
        return torch.exp(alpha + eps * beta.exp())

    def logpdf(self, eps, alpha, beta):
        """
        Compute the log-PDF of the distribution

        Arguments:
        ----------
        eps : torch.Tensor
            Noise sample
        alpha : torch.Tensor
            Mean parameters
        beta : torch.Tensor
            Standard deviation parameters (in log-scale)
        """
        z = self.g(eps, alpha, beta)
        sigma = beta.exp()
        log_phi = self.norm.log_prob((z.log() - alpha) / sigma)
        return torch.sum(log_phi - sigma.log() - z.log())

    def mode(self, alpha, beta):
        """
        Compute the mode of the distribution

        Arguments:
        ----------
        alpha : torch.Tensor
            Mean parameters
        beta : torch.Tensor
            Standard deviation parameters (in log-scale)
        """
        return torch.exp(alpha - beta.exp() ** 2)

    def mean(self, alpha, beta):
        """
        Compute the mean of the distribution

        Arguments:
        ----------
        alpha : torch.Tensor
            Mean parameters
        beta : torch.Tensor
            Standard deviation parameters (in log-scale)
        """
        return torch.exp(alpha + 0.5 * beta.exp() ** 2)

    def median(self, alpha, beta):
        """
        Compute the median of the distribution

        Arguments:
        ----------
        alpha : torch.Tensor
            Mean parameters
        beta : torch.Tensor
            Standard deviation parameters (in log-scale)
        """
        return alpha.exp()

    def variance(self, alpha, beta):
        """
        Compute the variance of the distribution

        Arguments:
        ----------
        alpha : torch.Tensor
            Mean parameters
        beta : torch.Tensor
            Standard deviation parameters (in log-scale)
        """
        sigma2 = beta.exp() ** 2
        return (sigma2.exp() - 1) * torch.exp(2 * alpha + sigma2)

    def pdf(self, x, alpha, beta):
        """
        Evaluate the PDF of the distribution at all values in `x`

        Arguments:
        ----------
        x : torch.Tensor
            Values to evaluate the distribution at
        alpha : torch.Tensor
            Mean parameters
        beta : torch.Tensor
            Standard deviation parameters (in log-scale)
        """
        sigma2 = beta.exp() ** 2
        return torch.exp(-((torch.log(x) - alpha) ** 2) / (2 * sigma2)) / (x * torch.sqrt(sigma2 * 2 * np.pi))


class TruncatedNormal(Posterior):
    """
    Truncated Normal posterior distribution
    """

    def __init__(self, bounds):
        """
        Initialize the distribution with fixed trunacation bounds

        Arguments:
        ----------
        bounds : iterable (of length two)
            The two min-max bounds of the truncated normal distribution
        """
        self.bounds = torch.tensor(bounds, dtype=torch.float64, requires_grad=False)
        self.norm = torch.distributions.Normal(loc=0.0, scale=1.0)
        self.val = None

    def sample_epsilon(self, size):
        """
        Sample noise
        """
        eps_arr = np.random.uniform(low=0.0, high=1.0, size=size)
        return torch.tensor(eps_arr, dtype=torch.float64, requires_grad=False)

    def bounds_rescaled(self, alpha, beta, sigma=None):
        """
        Rescale the bounds to compute the CDF

        Arguments:
        ----------
        alpha : torch.Tensor
            Mean parameters
        beta : torch.Tensor
            Standard deviation parameters (in log-scale)
        sigma : torch.Tensor (optional, default: None)
            Standard deviation parameters in linear scale, will be computed if not provided
        """
        if sigma is None:
            sigma = beta.exp()
        return ((self.bounds - alpha.unsqueeze(1)) / sigma.unsqueeze(1))

    def g(self, eps, alpha, beta, return_bounds_cdf_diff=False):
        """
        Reparameterize the posterior for a given set of parameters and noise sample

        Arguments:
        ----------
        eps : torch.Tensor
            Noise sample
        alpha : torch.Tensor
            Mean parameters
        beta : torch.Tensor
            Standard deviation parameters (in log-scale)
        """
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
        """
        Compute the log-PDF of the distribution

        Arguments:
        ----------
        eps : torch.Tensor
            Noise sample
        alpha : torch.Tensor
            Mean parameters
        beta : torch.Tensor
            Standard deviation parameters (in log-scale)
        """
        z, bound_cdf_diff = self.g(eps, alpha, beta, True)
        log_num = self.norm.log_prob((z - alpha) / beta.exp())
        log_denom = beta + bound_cdf_diff.log()
        logpdf = log_num - log_denom
        return logpdf.sum()

    def pdf(self, x, alpha, beta):
        """
        Compute the PDF of the distribution

        Arguments:
        ----------
        x : torch.Tensor
            Values to evaluate the posterior at
        alpha : torch.Tensor
            Mean parameters
        beta : torch.Tensor
            Standard deviation parameters (in log-scale)
        """
        bounds_rescaled = self.bounds_rescaled(alpha, beta)
        return truncnorm.pdf(a=bounds_rescaled[:, 0].detach().numpy(),
                             b=bounds_rescaled[:, 1].detach().numpy(),
                             loc=alpha.detach().numpy(), scale=beta.exp().detach().numpy(),
                             x=x.numpy())

    def mode(self, alpha, beta):
        """
        Compute the mode of the distribution

        Arguments:
        ----------
        alpha : torch.Tensor
            Mean parameters
        beta : torch.Tensor
            Standard deviation parameters (in log-scale)
        """
        return torch.clamp(alpha.detach(), self.bounds[0], self.bounds[1]).numpy()

    def mean(self, alpha, beta):
        """
        Compute the mean of the distribution

        Arguments:
        ----------
        alpha : torch.Tensor
            Mean parameters
        beta : torch.Tensor
            Standard deviation parameters (in log-scale)
        """
        bounds_rescaled = self.bounds_rescaled(alpha, beta)
        return truncnorm.mean(a=bounds_rescaled[:, 0].detach().numpy(),
                              b=bounds_rescaled[:, 1].detach().numpy(),
                              loc=alpha.detach().numpy(), scale=beta.exp().detach().numpy())

    def median(self, alpha, beta):
        """
        Compute the median of the distribution

        Arguments:
        ----------
        alpha : torch.Tensor
            Mean parameters
        beta : torch.Tensor
            Standard deviation parameters (in log-scale)
        """
        bounds_rescaled = self.bounds_rescaled(alpha, beta)
        return truncnorm.median(a=bounds_rescaled[:, 0].detach().numpy(),
                                b=bounds_rescaled[:, 1].detach().numpy(),
                                loc=alpha.detach().numpy(), scale=beta.exp().detach().numpy())

    def variance(self, alpha, beta):
        """
        Compute the variance of the distribution

        Arguments:
        ----------
        alpha : torch.Tensor
            Mean parameters
        beta : torch.Tensor
            Standard deviation parameters (in log-scale)
        """
        bounds_rescaled = self.bounds_rescaled(alpha, beta)
        return truncnorm.var(a=bounds_rescaled[:, 0].detach().numpy(),
                             b=bounds_rescaled[:, 1].detach().numpy(),
                             loc=alpha.detach().numpy(), scale=beta.exp().detach().numpy())
