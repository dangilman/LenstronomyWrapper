import numpy as np
from scipy.interpolate import interp1d

def pdf_from_samples(samples, nbins):

    hist, bins = np.histogram(samples, bins=nbins)
    bin_midpoints = bins[:-1] + np.diff(bins) / 2

    max_hist = float(np.max(hist))

    hist = np.array(hist)/max_hist ** -1

    pdf_interp = interp1d(bin_midpoints, hist, bounds_error=False,
                          fill_value=0.)

    return pdf_interp

def sample_distribution(samples, nbins, weights, nsamples):

    hist, bins = np.histogram(samples, bins=nbins, weights=weights)
    bin_midpoints = bins[:-1] + np.diff(bins) / 2
    cdf = np.cumsum(hist)

    if cdf[-1] == 0:
        return None

    cdf = cdf / cdf[-1]
    values = np.random.rand(nsamples)
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = bin_midpoints[value_bins]

    return random_from_cdf
