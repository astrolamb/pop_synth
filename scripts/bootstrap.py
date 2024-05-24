"""
a python script to bootstrap uncertainties on mean, variance, skewness and
kurtosis on our simulated GWB
"""

from argparse import ArgumentParser
import numpy as np
from scipy import stats

def bootstrap(data):
    """
    bootstrap the data

    Parameters
    ----------
    data: array
        the data to bootstrap

    Returns
    -------
    data: array
        the bootstrapped data
    """
    n = len(data)
    idx = np.random.randint(0, n, n)
    return data[idx]

def compute_stats(data):
    """
    compute the mean, variance, skewness and kurtosis of the data as a function
    of frequency

    Parameters
    ----------
    data: array
        the data to compute the statistics

    Returns
    -------
    mean: float
        the mean of the data
    variance: float
        the variance of the data
    skewness: float
        the skewness of the data
    kurtosis: float
        the kurtosis of the data
    """
    mu = np.mean(data, axis=0)
    var = np.var(data, axis=0)
    skew = stats.skew(data, axis=0)
    kurt = stats.kurtosis(data, axis=0)
    return mu, var, skew, kurt

if __name__ == '__main__':

    # parse the command line arguments
    parser = ArgumentParser()
    parser.add_argument('--model_idx', type=int,
                        help='the model index to bootstrap')
    args = parser.parse_args()

    h2cf = np.load(f'./data/accre_runs/h2cf_model{args.model_idx}.npy')

    # bootstrap the data
    h2cf = bootstrap(h2cf)

    # compute the statistics
    mean, variance, skewness, kurtosis = compute_stats(h2cf)

    # save the statistics as a structured array
    stats = np.zeros(len(mean), dtype=[('mean', float),
                                       ('variance', float),
                                       ('skewness', float),
                                       ('kurtosis', float)])
    stats['mean'] = mean
    stats['variance'] = variance
    stats['skewness'] = skewness
    stats['kurtosis'] = kurtosis

    np.save(f'./data/accre_runs/h2cf_model{args.model_idx}_stats.npy', stats)
