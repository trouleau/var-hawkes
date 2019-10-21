import numpy as np

import torch

from . import models
from . import hawkes_model
from . import priors


class VariationalInferenceLearner(object):

    def __init__(self, model, optimizer, tol=1e-5, lr_gamma=0.9999, max_iter=10000,
                 hyperparam_interval=100, hyperparam_offset=0, hyperparam_momentum=0.5, debug=False):
        if not isinstance(model, models.ModelHawkesVariational):
            raise ValueError("`model` should be a `ModelHawkesVariational` object")
        self.model = model
        self.optimizer = optimizer
        self.lr_gamma = lr_gamma
        self.tol = tol
        self.max_iter = int(max_iter)
        self.hyperparam_interval = hyperparam_interval
        self.hyperparam_offset = hyperparam_offset
        self.hyperparam_momentum = hyperparam_momentum
        self.debug = debug
        # Attributes
        self.coeffs = None
        self.coeffs_prev = None

    def _set_data(self, events, end_time):
        if not isinstance(events[0], list) and isinstance(events[0][0], np.ndarray):
            raise TypeError('Invalid `events` provided')
        self.dim = len(events)
        self.model.set_data(events, end_time)

    def _check_convergence(self):

        if torch.abs(self.coeffs - self.coeffs_prev).max() < self.tol:
            return True
        self.coeffs_prev = self.coeffs.detach().clone()
        return False

    def fit(self, events, end_time, x0, seed=None, callback=None):
        if seed:
            np.random.seed(seed)
        self._set_data(events, end_time)
        # Initialize estimate
        self.coeffs = x0.clone().detach().requires_grad_(True)
        self.coeffs_prev = self.coeffs.detach().clone()
        # Reset optimizer
        self.optimizer = type(self.optimizer)([self.coeffs], **self.optimizer.defaults)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=self.lr_gamma)
        for t in range(self.max_iter):
            self._n_iter_done = t
            
            # Gradient update
            self.optimizer.zero_grad()
            val = -1.0 * self.model.objective(self.coeffs)
            val.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Check that the optimization did not fail
            if torch.isnan(self.coeffs).any():
                raise ValueError('NaNs in coeffs! Stop optimization...')
            # Convergence check
            if self._check_convergence():
                if callback:  # Callback before the end
                    callback(self, end='\n')
                    print('Converged!')
                break
            elif callback:  # Callback at each iteration
                callback(self)
            # Update hyper-parameters
            if (t+1) % self.hyperparam_interval == 0 and t > self.hyperparam_offset:
                self.model.hyper_parameter_learn(self.coeffs.detach(),
                                                 momentum=self.hyperparam_momentum)
        return self.coeffs


class MLELearner(object):

    def __init__(self, hawkesmodel, prior, optimizer, tol=1e-4, max_iter=10000,
                debug=False):
        if not isinstance(hawkesmodel, hawkes_model.HawkesModel):
            raise ValueError("`model` should be a `HawkesModel` object")
        if not isinstance(prior, priors.Prior):
            raise ValueError("`prior` should be a `priors` object")
        self.model = hawkesmodel
        self.prior = prior
        self.optimizer = optimizer
        self.tol = tol
        self.max_iter = int(max_iter)
        self.debug = debug
        # Attributes
        self.coeffs = None
        self.coeffs_prev = None

    def _set_data(self, events, end_time):
        if not isinstance(events[0], list) and isinstance(events[0][0], np.ndarray):
            raise TypeError('Invalid `events` provided')
        self.dim = len(events)
        self.model.set_data(events, end_time)

    def _check_convergence(self):
        if torch.abs(self.coeffs - self.coeffs_prev).max() < self.tol:
            return True
        self.coeffs_prev = self.coeffs.detach().clone()
        return False

    def fit(self, events, end_time, x0, seed=None, callback=None):
        if seed:
            np.random.seed(seed)
        self._set_data(events, end_time)
        # Initialize estimate
        self.coeffs = x0.clone().detach().requires_grad_(True)
        self.coeffs_prev = self.coeffs.detach().clone()
        # Reset optimizer
        self.optimizer = type(self.optimizer)([self.coeffs], **self.optimizer.defaults)
        for t in range(self.max_iter):
            self._n_iter_done = t
            
            # Gradient update
            self.optimizer.zero_grad()
            mu = self.coeffs[:self.dim]
            W = self.coeffs[self.dim:].reshape(self.dim, self.dim, self.model.excitation.M)
            loss = -1.0 * self.model.log_likelihood(mu, W) - 1.0 * self.prior.logprior(self.coeffs)
            
            loss.backward()
            self.optimizer.step()
            self.coeffs.requires_grad = False
            self.coeffs.abs_()
            self.coeffs.requires_grad = True
            # Check that the optimization did not fail
            if torch.isnan(self.coeffs).any():
                raise ValueError('NaNs in coeffs! Stop optimization...')
            # Convergence check
            if self._check_convergence():
                if callback:  # Callback before the end
                    callback(self, end='\n')
                    print(f'loss = {loss}','Converged!')
                break
            elif callback:  # Callback at each iteration
                callback(self)
        print(loss)
        return self.coeffs
