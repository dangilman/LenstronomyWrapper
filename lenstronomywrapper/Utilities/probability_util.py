import numpy as np
from scipy.interpolate import interp1d

def confidence_intervals_from_pdf(pdf_function, sample_min, sample_max, num_sigma, ndraws=10000):

    samples = []
    while len(samples) < ndraws:
        u = np.random.rand()
        value = np.random.uniform(sample_min, sample_max)
        if pdf_function(value) > u:
            samples.append(value)

    return confidence_intervals_from_samples(samples, num_sigma)

def confidence_intervals_from_samples(sample, num_sigma):

    """
    computes the upper and lower sigma from the median value.
    This functions gives good error estimates for skewed pdf's
    :param sample: 1-D sample
    :return: median, lower_sigma, upper_sigma
    """

    if num_sigma > 3:
        raise ValueError("Number of sigma-constraints restricted to three. %s not valid" % num_sigma)
    num = len(sample)
    median = np.median(sample)
    sorted_sample = np.sort(sample)

    num_threshold1 = int(round((num-1)*0.841345))
    num_threshold2 = int(round((num-1)*0.977249868))
    num_threshold3 = int(round((num-1)*0.998650102))

    if num_sigma == 1:
        upper_sigma1 = sorted_sample[num_threshold1 - 1]
        lower_sigma1 = sorted_sample[num - num_threshold1 - 1]
        return median, [median-lower_sigma1, upper_sigma1-median]
    if num_sigma == 2:
        upper_sigma2 = sorted_sample[num_threshold2 - 1]
        lower_sigma2 = sorted_sample[num - num_threshold2 - 1]
        return median, [median-lower_sigma2, upper_sigma2-median]

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
