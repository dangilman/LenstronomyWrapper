import numpy as np


def sample_distribution(samples, nbins, weights, nsamples):

    hist, bins = np.histogram(samples, bins=nbins, weights=weights)
    bin_midpoints = bins[:-1] + np.diff(bins) / 2
    cdf = np.cumsum(hist)

    cdf = cdf / cdf[-1]
    values = np.random.rand(nsamples)
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = bin_midpoints[value_bins]

    return random_from_cdf
