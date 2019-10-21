import numpy as np
import argparse
import json
import sys
import os

# External libraries
from tick.hawkes.simulation import SimuHawkesExpKernels
from tick.hawkes.inference import HawkesADM4, HawkesSumGaussians
import torch
from torch import optim

# Internal libraries
from varhawkes import models
from varhawkes import posteriors
from varhawkes import priors
from varhawkes import hawkes_model, excitation_kernels
from varhawkes import learners
from varhawkes import utils
from varhawkes.utils import metrics

THRESHOLD = 0.03


def make_object(module, name, args):
    return getattr(module, name)(**args)


class LearnerCallback:
    
    def __init__(self, x0, adjacency_true, acc_thresh=0.01, print_every=10, line_return=False):
        self.print_every = print_every
        self.last_coeffs = x0.clone()
        self.adjacency_true = adjacency_true.ravel()
        self.acc_thresh = acc_thresh
        self.default_end = "\n" if line_return else ""

    def __call__(self, learner_obj, end=None):
        t = learner_obj._n_iter_done + 1
        if t % self.print_every == 0:
            # Extract number of parameters
            n_nodes = learner_obj.dim
            n_params = learner_obj.model.n_params  # n_nodes * (learner_obj.model.excitation.M*n_nodes + 1)
            # Split parameters
            xt = learner_obj.coeffs.detach()
            alpha, beta = xt[:n_params], xt[n_params:]
            # Compute mode estimator
            z = learner_obj.model.posterior.mode(alpha, beta)
            
            x_diff = torch.abs(self.last_coeffs - xt).max()
            self.last_coeffs = xt.clone()
            adj_learned = z[n_nodes:].cpu().numpy().reshape(n_nodes, n_nodes, learner_obj.model.model.excitation.M).sum(2).flatten()
            f1score = metrics.fscore(adj_learned, self.adjacency_true, self.acc_thresh, 1.0)
            precat20 = metrics.precision_at_n(adj_learned, self.adjacency_true, n=20)
            precat50 = metrics.precision_at_n(adj_learned, self.adjacency_true, n=50)
            precat100 = metrics.precision_at_n(adj_learned, self.adjacency_true, n=100)
            relerr = metrics.relerr(adj_learned, self.adjacency_true)
            
            end = end = self.default_end if end is None else end
            print(f"\riter: {t:>5d} | f1-score: {f1score:.2f} | relerr: {relerr:.3f} | p@20-50-100: {precat20:.2f} {precat50:.2f} {precat100:.2f} | "
                  f"dx: {x_diff:.2e}"
                  "    ", end=end, flush=True)


def generate_data(adjacency, decay, baseline, n_jumps, sim_seed=None, verbose=False):
    # Set random seed (for reproducibility)
    if sim_seed is None:
        sim_seed = np.random.randint(2**31)
    # Simulate data
    hawkes_simu = SimuHawkesExpKernels(adjacency=adjacency, decays=decay, baseline=baseline,
                                       max_jumps=n_jumps, seed=sim_seed, verbose=verbose)
    hawkes_simu.simulate()
    return hawkes_simu.timestamps, hawkes_simu.end_time, sim_seed


def learn_adm4(events, end_time, return_learner=False, verbose=False, **kwargs):
    learner_mle = HawkesADM4(**kwargs, verbose=verbose)
    learner_mle.fit(events, end_time)
    if return_learner:
        return learner_mle
    return learner_mle.baseline, learner_mle.adjacency


def learn_sum_gaussians(events, end_time, return_learner=False, verbose=False, **kwargs):
    learner_mle = HawkesSumGaussians(**kwargs, verbose=verbose)
    learner_mle.fit(events, end_time)
    if return_learner:
        return learner_mle
    return learner_mle.baseline, learner_mle.amplitudes


def learn_vi(events, end_time, vi_seed, adjacency_true, inference_param_dict, return_learner=False):
    # Extract some parameters for easier access
    n_nodes = len(events)
    M = inference_param_dict['excitation']['args'].get('M', 1)
    n_params = n_nodes * (n_nodes * M + 1)
    n_edges = M * n_nodes ** 2
    # Set seed
    np.random.seed(vi_seed)
    # Set starting pointM * n_nodes ** 2
    x0 = torch.tensor(
        np.hstack((
            np.hstack((  # alpha, the mean of the parameters
                np.random.normal(loc=0.1, scale=0.1, size=n_nodes),
                np.random.normal(loc=0.1, scale=0.1, size=n_edges),)),
            np.hstack((  # beta=log(sigma), log of the variance of the parameters
                np.log(np.clip(np.random.normal(loc=0.2, scale=0.1, size=n_nodes), 1e-1, 2.0)),
                np.log(np.clip(np.random.normal(loc=0.2, scale=0.1, size=n_edges), 1e-1, 2.0)),))
        )),
        dtype=torch.float64, requires_grad=True
    )
    # Init Hawkes process model object
    excitation_obj = make_object(excitation_kernels, **inference_param_dict['excitation'])
    hawkes_model_obj = hawkes_model.HawkesModel(excitation=excitation_obj, verbose=False)
    # Init the posterior object
    posterior_obj = make_object(posteriors, **inference_param_dict['posterior'])
    # Init the prior object
    prior_type = inference_param_dict['prior']['name']
    prior_args = inference_param_dict['prior']['args']
    prior_args['C'] = torch.tensor(prior_args['C'], dtype=torch.float64)  # cast to tensor
    prior_obj = make_object(priors, prior_type, prior_args)
    # Init the variational inference model object
    model = models.ModelHawkesVariational(
        model=hawkes_model_obj, posterior=posterior_obj, prior=prior_obj,
        **inference_param_dict['model']['args'])
    # Init callback object (for monitoring purposes)
    callback = LearnerCallback(x0=x0.detach(), adjacency_true=adjacency_true,
                               acc_thresh=THRESHOLD, print_every=100, line_return=False)
    # Init the optimizer
    opt_type = inference_param_dict['optimizer']['name']
    opt_args = inference_param_dict['optimizer']['args']
    opt = getattr(optim, opt_type)([x0], **opt_args)
    # Init learner
    learner = learners.VariationalInferenceLearner(
        model=model, optimizer=opt, **inference_param_dict['learner']['args'])
    # Fit the model
    events_t = [torch.tensor(events_i) for events_i in events]  # cast to tensor
    learner.fit(events_t, end_time, x0, callback=callback)
    print()
    if return_learner:
        return learner
    # Extract the mode of the posterior
    z_est_mode = learner.model.posterior.mode(learner.coeffs[:n_params], learner.coeffs[n_params:])
    adj_est = z_est_mode[n_nodes:].detach().numpy()
    adj_est = np.reshape(adj_est, (n_nodes, n_nodes, M)).sum(-1).ravel()
    coeffs_est = learner.coeffs.detach().numpy()
    return coeffs_est, adj_est


def run(exp_dir, param_filename, output_filename, stdout=None, stderr=None):
    # Reset random seed
    np.random.seed(None)

    if stdout is not None:
        sys.stdout = open(stdout, 'w')
    if stderr is not None:
        sys.stderr = open(stderr, 'w')

    print('\nExperiment parameters')
    print('=====================')
    print(f'        exp_dir = {exp_dir:s}')
    print(f' param_filename = {param_filename:s}')
    print(f'output_filename = {output_filename:s}')
    print(flush=True)

    # Load parameters from file
    param_filename = os.path.join(exp_dir, param_filename)
    if not os.path.exists(param_filename):
        raise FileNotFoundError(
            'Input file `{:s}` not found.'.format(param_filename))
    with open(param_filename, 'r') as param_file:
        param_dict = json.load(param_file)

    result_dict = {}

    print('\nSIMULATION')
    print('==========')
    print(flush=True)

    n_jumps = param_dict['simulation']['n_jumps']
    adjacency = np.array(param_dict['simulation']['adjacency'])
    decay = param_dict['simulation']['decay']
    baseline = np.array(param_dict['simulation']['baseline'])
    sim_seed = param_dict['simulation'].get('sim_seed')

    print('\ndecay:')
    print(decay)
    print('baseline:')
    print(baseline.round(2))
    print('adjacency:')
    print(adjacency.round(2))

    print('\nGenerate data...')
    # Simulate data
    events, end_time, sim_seed = generate_data(adjacency=adjacency, decay=decay, baseline=baseline,
                                               n_jumps=n_jumps, sim_seed=sim_seed)
    n_jumps_per_dim = list(map(len, events))
    print(f'simulation random seed: {sim_seed}')
    print()
    print('Number of jumps:', sum(n_jumps_per_dim))
    print('per node:', n_jumps_per_dim)

    result_dict.update({
        'sim_seed': sim_seed,  # Simulation random seed
        'n_jumps': n_jumps,    # Number of arrivals
    })

    print('\nINFERENCE')
    print('=========')

    for key, inference_param_dict in param_dict['inference'].items():

        if key == 'adm4':
            print('\nRun ADM4')
            print('-------')
            print()
            baseline_adm4, adj_adm4 = learn_adm4(events, end_time, **inference_param_dict)
            adj_adm4 = adj_adm4.ravel()
            print('  relerr:', utils.metrics.relerr(adj_adm4, adjacency.ravel()))
            print('      f1:', utils.metrics.fscore(adj_adm4, adjacency.ravel(),
                                                    threshold=THRESHOLD, beta=1.0))
            print(' prec@20:', utils.metrics.precision_at_n(adj_adm4, adjacency.ravel(), n=20))
            print(' prec@50:', utils.metrics.precision_at_n(adj_adm4, adjacency.ravel(), n=50))
            print('prec@100:', utils.metrics.precision_at_n(adj_adm4, adjacency.ravel(), n=100))

            result_dict.update({
                key: {
                    'baseline': baseline_adm4.tolist(),  # ADM4 Baseline Estimator
                    'adjacency': adj_adm4.tolist(),      # ADM4 Adjacencey Estimator
                }
            })

        if key == 'sum_gaussians':
            print('\nRun Sum-Gaussians')
            print('-------')
            print()
            baseline_sumgauss, adj_sumgauss = learn_sum_gaussians(
                events, end_time, **inference_param_dict)
            adj_sumgauss = adj_sumgauss.sum(axis=-1).ravel()
            print('  relerr:', utils.metrics.relerr(adj_sumgauss, adjacency.ravel()))
            print('      f1:', utils.metrics.fscore(adj_sumgauss, adjacency.ravel(),
                                                    threshold=THRESHOLD, beta=1.0))
            print(' prec@20:', utils.metrics.precision_at_n(adj_sumgauss, adjacency.ravel(), n=20))
            print(' prec@50:', utils.metrics.precision_at_n(adj_sumgauss, adjacency.ravel(), n=50))
            print('prec@100:', utils.metrics.precision_at_n(adj_sumgauss, adjacency.ravel(), n=100))

            result_dict.update({
                key: {
                    'baseline': baseline_sumgauss.tolist(),  # sum_gaussians Baseline Estimator
                    'adjacency': adj_sumgauss.tolist(),      # sum_gaussians Adjacencey Estimator
                }
            })

        if key.startswith('vi'):
            print(f'\nRun VI ({key:s})')
            print('------')
            # Set random seed (for reproducibility)
            np.random.seed()  # Reset random number generator to avoid dependency on simulation seed
            vi_seed = np.random.randint(2**32 - 1)
            print(f'vi random seed: {vi_seed}')
            # Run inference
            coeffs_var, adj_var = learn_vi(events, end_time, vi_seed, adjacency.ravel(),
                                           inference_param_dict)
            adj_var = adj_var.ravel()
            print()
            print('  relerr:', utils.metrics.relerr(adj_var, adjacency.ravel()))
            print('      f1:', utils.metrics.fscore(adj_var, adjacency.ravel(),
                                                    threshold=THRESHOLD, beta=1.0))
            print(' prec@20:', utils.metrics.precision_at_n(adj_var, adjacency.ravel(), n=20))
            print(' prec@50:', utils.metrics.precision_at_n(adj_var, adjacency.ravel(), n=50))
            print('prec@100:', utils.metrics.precision_at_n(adj_var, adjacency.ravel(), n=100))

            result_dict.update({
                key: {
                    'vi_seed': vi_seed,             # VI random seed
                    'coeffs': coeffs_var.tolist(),  # VI parameters
                    'adjacency': adj_var.tolist(),  # VI Estimator
                }
            })

    print('\n\nSave results...')

    with open(os.path.join(exp_dir, output_filename), 'w') as output_file:
        json.dump(result_dict, output_file)

    # Log that the run is finished
    print('\n\nFinished.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', dest='dir', type=str,
                        required=True, help="Working directory")
    parser.add_argument('-p', '--params', dest='param_filename', type=str,
                        required=False, default='params.json',
                        help="Input parameter file (JSON)")
    parser.add_argument('-o', '--outfile', dest='output_filename', type=str,
                        required=False, default='output.json',
                        help="Output file (JSON)")
    args = parser.parse_args()

    run(exp_dir=args.dir, param_filename=args.param_filename,
        output_filename=args.output_filename)
