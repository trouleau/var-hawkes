"""
Performance metrics
"""
import numpy as np
import scipy


def edge_error(adj_excit_test, adj_excit_true, delta_t, n_nodes):
    diff_excit = (adj_excit_test - adj_excit_true).abs()
    return diff_excit.sum() * delta_t / n_nodes**2


def AUC(adj_test, adj_true):
    threshold_range = np.linspace(adj_test.max() + 0.1, 0.0, num=50)
    tps = np.array([utils.metrics.true_positive(adj_test, adj_true, threshold=x) for x in threshold_range])
    tns = np.array([utils.metrics.true_negative(adj_test, adj_true, threshold=x) for x in threshold_range])
    fns = np.array([utils.metrics.false_negative(adj_test, adj_true, threshold=x) for x in threshold_range])
    fps = np.array([utils.metrics.false_positive(adj_test, adj_true, threshold=x) for x in threshold_range])
    tpr = tps/(tps+fns)
    fpr = fps/(tns+fps)
    AUC_val = sum(np.diff(fpr) * tpr[1:])
    return AUC_val


def accuracy(adj_test, adj_true, threshold=0.05):
    assert (len(adj_test.shape) == 1) and (len(adj_true.shape) == 1), \
     "Parameters should be one-dimensional"
    shape = len(adj_test)
    n_err = np.sum((adj_test >= threshold) ^ (adj_true > 0))
    return (shape - n_err) / shape


def mean_kendall_rank_corr(adj_est, adj_true):
    assert (len(adj_est.shape) == 1) and (len(adj_true.shape) == 1), \
         "Parameters should be one-dimensional"
    dim = int(np.sqrt(adj_est.shape[0]))
    adj_est = np.reshape(adj_est, (dim, dim))
    adj_true = np.reshape(adj_true, (dim, dim))
    return np.mean([scipy.stats.kendalltau(adj_est[:, i], adj_true[:, i]).correlation
                    for i in range(dim)])


def precision_at_n(adj_test, adj_true, n):
    sorted_args = np.argsort(adj_test)[::-1]
    return np.sum(adj_true[sorted_args][:n] > 0) / n


def true_positive(adj_test, adj_true, threshold=0.05):
    assert (len(adj_test.shape) == 1) and (len(adj_true.shape) == 1), \
     "Parameters should be one-dimensional"
    return np.sum((adj_test >= threshold) * (adj_true > 0))


def false_positive(adj_test, adj_true, threshold=0.05):
    assert (len(adj_test.shape) == 1) and (len(adj_true.shape) == 1), \
     "Parameters should be one-dimensional"
    return np.sum((adj_test >= threshold) * (adj_true == 0))


def false_negative(adj_test, adj_true, threshold=0.05):
    assert (len(adj_test.shape) == 1) and (len(adj_true.shape) == 1), \
     "Parameters should be one-dimensional"
    return np.sum((adj_test < threshold) * (adj_true > 0))


def true_negative(adj_test, adj_true, threshold=0.05):
    assert (len(adj_test.shape) == 1) and (len(adj_true.shape) == 1), \
     "Parameters should be one-dimensional"
    return np.sum((adj_test < threshold) * (adj_true == 0))


def tp(adj_test, adj_true, threshold=0.05):
    return true_positive(adj_test, adj_true, threshold)


def fp(adj_test, adj_true, threshold=0.05):
    return false_positive(adj_test, adj_true, threshold)


def tn(adj_test, adj_true, threshold=0.05):
    return true_negative(adj_test, adj_true, threshold)


def fn(adj_test, adj_true, threshold=0.05):
    return false_negative(adj_test, adj_true, threshold)


def recall(adj_test, adj_true, threshold=0.05):
    return tp(adj_test, adj_true, threshold) / np.sum(adj_true > 0)


def precision(adj_test, adj_true, threshold=0.05):
    return tp(adj_test, adj_true, threshold) / np.sum(adj_test > threshold)


def fscore(adj_test, adj_true, threshold=0.05, beta=1.0):
    rec_val = recall(adj_test, adj_true, threshold)
    prec_val = precision(adj_test, adj_true, threshold)
    return (1 + beta ** 2) * prec_val * rec_val / (beta ** 2 * prec_val + rec_val)


def tpr(adj_test, adj_true, threshold=0.05):
    return recall(adj_test, adj_true, threshold)


def fpr(adj_test, adj_true, threshold=0.05):
    return fp(adj_test, adj_true, threshold) / np.sum(adj_true == 0)


def nrmse(adj_test, adj_true):
    return np.sqrt(np.sum((adj_test - adj_true) ** 2)) / np.sum(adj_true ** 2)


def relerr(adj_test, adj_true):
    mask = adj_true > 0
    n_nodes = adj_true.shape[0]
    rel_err = np.sum(np.abs(adj_test - adj_true)[mask] / adj_true[mask]) + np.abs(adj_test - adj_true)[~mask].sum() / adj_true[mask].min()
    return rel_err / n_nodes
