import numpy as np

import torch

from tick.hawkes.model import (
    ModelHawkesExpKernLogLik as _ModelHawkesExpKernLogLik
)

from .utils import softmax
from .posteriors import Posterior
from .priors import Prior
from .hawkes_model import HawkesModel


class ModelHawkesVariational:

    def __init__(self, model, posterior, prior, n_samples, n_weights, weight_temp=1, device='cpu'):
        """
        Initialize the model

        Arguments:
        ----------
        model : HawkesModel
            Hawkes process object that implements the log-likelihood
        posterior : Posterior
            Posterior object
        prior : Prior
            Prior object
        n_samples : int
            Number of samples used fort he Monte Carlo estimate of expectations
        n_weights : int
            Number of samples used for the importance weighted posterior
        C : float
            Inverse weight of the regularizer
        weight_temp : float
            Tempering weight of the importance weights
        """
        if not isinstance(model, HawkesModel):
            raise ValueError("`model` should be a `HawkesModel` object")
        self.model = model
        if not isinstance(posterior, Posterior):
            raise ValueError("`posterior` should be a `Posterior` object")
        self.posterior = posterior
        if not isinstance(prior, Prior):
            raise ValueError("`prior` should be a `Prior` object")
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.prior = prior
        self.n_samples = n_samples
        self.n_weights = n_weights
        self.weight_temp = weight_temp
        self.n_jumps = None
        self.dim = None
        self.n_params = None
        self.n_var_params = None
        self.alpha = None
        self.beta = None

    def set_data(self, events, end_time):
        """
        Set the data for the model
        """
        events = [num.to(self.device) for num in events]  # Moving the tensors to GPU if available
        # Set the model object
        self.model.set_data(events, end_time)
        # Set various util attributes
        self.dim = len(events)
        self.n_jumps = sum(map(len, events))
        self.n_params = self.dim * (self.model.excitation.M * self.dim + 1)
        self.n_var_params = 2 * self.n_params

    def _log_importance_weight(self, eps, alpha, beta):
        """
        Compute the value of a single importance weight `log(w_i)`
        """
        # Reparametrize the variational parameters
        z = self.posterior.g(eps, alpha, beta)
        # Split and reshape parameters
        mu = z[:self.dim]
        W = z[self.dim:].reshape(self.dim, self.dim, self.model.excitation.M)
        # Compute log-posterior
        logpost = self.posterior.logpdf(eps, alpha, beta)
        # Evaluate the log-likelihood
        loglik = self.model.log_likelihood(mu, W)
        # Compute the log-prior
        logprior = self.prior.logprior(z)
        return loglik + logprior - logpost

    def _objective_l(self, eps_l, alpha, beta):
        log_w_arr = torch.zeros(self.n_weights, dtype=torch.float64)
        for i in range(self.n_weights):
            eps = eps_l[i]
            # Compute the importance weights (and their gradients)
            log_w_arr[i] = self._log_importance_weight(eps, alpha, beta)
        # Temper the weights
        log_w_arr /= self.weight_temp
        w_tilde = softmax(log_w_arr).detach()  # Detach `w_tilde` from backward computations
        # Compute the weighted average over all `n_weights` samples
        value_i = w_tilde * log_w_arr
        return value_i.sum()

    def objective(self, x, seed=None):
        """
        Importance weighted variational objective function

        Arguments:
        ----------
        x : torch.Tensor
             The parameters to optimize
        seed : int (optional)
            Random seed for samples
        """
        if seed:
            np.random.seed(seed)  # Sampling is done in posterior with numpy
        # Split the parameters into `alpha` and `beta`
        alpha = x[:self.n_params]
        beta = x[self.n_params:]
        # Sample noise
        sample_size = (self.n_samples, self.n_weights, self.n_params)
        eps_arr = self.posterior.sample_epsilon(size=sample_size)
        # Initialize the output variables
        value = 0.0
        # Compute a Monte Carlo estimate of the expectation
        for l in range(self.n_samples):
            value += self._objective_l(eps_arr[l],  alpha, beta)
        value /= self.n_samples
        return value

    def hyper_parameter_learn(self, x, momentum=0.5):
        """
        Learn the hyper parameters of the model
        """
        opt_C_now = torch.zeros((self.n_weights, self.n_params), dtype=torch.float64)
        log_w_arr = torch.zeros(self.n_weights, dtype=torch.float64)
        # Split the parameters into `alpha` and `beta`
        alpha = x[:self.n_params]
        beta = x[self.n_params:]
        # Sample noise
        sample_size = (self.n_weights, self.n_params)
        eps_arr = self.posterior.sample_epsilon(size=sample_size)
        for i in range(self.n_weights):
            eps = eps_arr[i]
            # Compute the importance weights (and their gradients)
            log_w_arr[i] = self._log_importance_weight(eps, alpha, beta)
            z_i = self.posterior.g(eps, alpha, beta)
            opt_C_now[i] = self.prior.opt_hyper(z_i)
        # Temper the weights
        log_w_arr /= self.weight_temp
        w_tilde = softmax(log_w_arr).detach()  # Detach `w_tilde` from backward computations
        # Compute the weighted average over all `n_weights` samples
        opt_C = torch.matmul(w_tilde.unsqueeze(0), opt_C_now).squeeze().to(self.device)
        self.prior.C = (1-momentum) * opt_C + momentum * self.prior.C
        # self.prior.C = np.clip(self.prior.C, 1e-5, 1e3)

    def _sample_from_expected_importance_weighted_distribution(self, eps_arr_l, alpha, beta):
        # Reparametrize the variational parameters
        log_w_arr = torch.zeros(self.n_weights, dtype=torch.float64)
        for i in range(self.n_weights):
            eps = eps_arr_l[i]
            # Compute the importance weights (and their gradients)
            log_w_arr[i] = self._log_importance_weight(eps, alpha, beta)
        # Sample one of the `z` in `z_arr` w.p. proportional to `softmax(log_w_arr)`
        j = np.random.multinomial(n=1, pvals=softmax(log_w_arr).detach().numpy()).argmax()
        return self.posterior.g(eps_arr_l[j], alpha, beta).detach().numpy()

    def expected_importance_weighted_estimate(self, alpha, beta, n_samples=None, seed=None):
        """
        Return the mean of the expected importance weighted distribution at the
        parameters `x`

        Arguments:
        ----------
        alpha : np.ndarray (shape: `n_params`)
            The value of the variational parameters
        beta : np.ndarray (shape: `n_params`)
            The value of the variational parameters
        n_sample : int (optional)
            The number of Montre Carlo samples to use (if None, then the
            default `n_samples` attribute will be used)
        seed : int (optional)
            Random seed generator for `numpy.random`
        """
        n_samples = n_samples or self.n_samples
        if seed:
            np.random.seed(seed)
        # Sample noise
        eps_arr_t = self.posterior.sample_epsilon(
            size=(n_samples, self.n_weights, self.n_params))
        # Compute a Monte Carlo estimate of the expectation
        value = np.zeros(self.n_params)
        for l in range(n_samples):
            value += self._sample_from_expected_importance_weighted_distribution(
                eps_arr_t[l], alpha, beta)
        value /= n_samples
        return value
