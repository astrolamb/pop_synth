"""
a python script to bootstrap uncertainties on mean, variance, skewness and
kurtosis on our simulated GWB
"""

from argparse import ArgumentParser
import numpy as np
from scipy import stats

def bootstrap(n):
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
    idxs = np.random.randint(0, n, n)
    return idxs

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
    parser.add_argument('--real_idx', type=int,
                        help='the idx of the realization')
    args = parser.parse_args()

    h2cf = np.load(f'./data/accre_runs/h2cf_model{args.model_idx}.npy')

    # bootstrap the data
    # save the statistics as a structured array with the same length as the
    # number of frequencies and number of bootstraps
    stat_array = np.zeros((100000, 30), dtype=[('mean', float),
                                                  ('variance', float),
                                                  ('skewness', float),
                                                  ('kurtosis', float)])
    for ii in range(100000):
        h2cf_boot = h2cf[bootstrap(100000)]

        # compute the statistics
        mean, variance, skewness, kurtosis = compute_stats(h2cf_boot)

        # store the statistics
        stat_array[ii]['mean'] = mean
        stat_array[ii]['variance'] = variance
        stat_array[ii]['skewness'] = skewness
        stat_array[ii]['kurtosis'] = kurtosis

        # save the statistics every 1000 iterations
        if ii % 1000 == 0:
            np.save(f'./data/accre_runs/h2cf_model{args.model_idx}_stats_{args.real_idx}.npy',
                    stat_array)

    np.save(f'./data/accre_runs/h2cf_model{args.model_idx}_stats_{args.real_idx}.npy',
            stat_array)
