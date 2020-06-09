import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator

def interp_2d_likelihood(x, y, nbins, param_range_1, param_range_2):

    param_min_1, param_max_1 = param_range_1[0], param_range_1[1]
    param_min_2, param_max_2 = param_range_2[0], param_range_2[1]
    weights, _, _ = np.histogram2d(x, y, nbins, range=((param_min_1, param_max_1),
                                                      (param_min_2, param_max_2)))

    samples1 = np.linspace(param_min_1, param_max_1, weights.shape[0])
    samples2 = np.linspace(param_min_2, param_max_2, weights.shape[0])

    points = (samples1, samples2)

    interp = RegularGridInterpolator(points, weights)

    return interp

def transform_pdf_from_samples_2d(samples, samples_reference, nbins,
                                  param_range_1, param_range_2):

    interp = interp_2d_likelihood(samples[0], samples[1], nbins, param_range_1, param_range_2)
    interp_ref = interp_2d_likelihood(samples_reference[0], samples_reference[1], nbins,
                                      param_range_1, param_range_2)

    def _func(xpoints, ypoints):
        shape0 = xpoints.shape
        coords = np.array([xpoints.ravel(), ypoints.ravel()])
        L = len(xpoints.ravel())
        rescale = []
        for i in range(0, L):
            try:
                likeref = interp_ref(coords[i][0], coords[i][1])
                like = interp(coords[i][0], coords[i][1])
                rescale.append(likeref/like)
            except:
                rescale.append(0)

        rescale = np.array(rescale).reshape(shape0)

        inds_nan = np.isnan(rescale)

        rescale[inds_nan] = 0
        return rescale

    return _func

def transform_pdf_from_samples(samples_reference, samples, nbins):

    param_min, param_max = np.min(samples), np.max(samples)
    href, bref = np.histogram(samples_reference, nbins,
                              range=(param_min, param_max))

    h, b = np.histogram(samples, nbins,
                              range=(param_min, param_max))

    weights = []
    for i in range(0, nbins):
        if h[i] == 0:
            weights.append(0)
        else:
            weights.append(href[i]/h[i])
    weights = np.array(weights)

    samples = np.linspace(param_min, param_max, nbins)
    pdf = pdf_from_samples(samples, nbins, weights)

    return pdf


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

def pdf_from_samples(samples, nbins, weights=None):

    hist, bins = np.histogram(samples, bins=nbins, weights=weights)
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

# cov1 = [[0.2, 0.1], [0.1, 0.2]]
# cov2 = [[0.5, 0.2], [0.2, 0.5]]
#
# samplesref = np.random.multivariate_normal([0, 0.], cov1, 1000)
# samples = np.random.multivariate_normal([0, 0], cov2, 1000)
#
# import matplotlib.pyplot as plt
#
# like_ref, pran1, pran2 = interp_2d_likelihood(samplesref[:,0], samplesref[:,1], 10)
# like, _, _ = interp_2d_likelihood(samples[:,0], samples[:,1], 10,
#                                   param_range_1=pran1, param_range_2=pran2)
#
# xran, yran = np.linspace(pran1[0], pran1[1], 50), np.linspace(pran2[0], pran2[1], 50)
# xx, yy = np.meshgrid(xran, yran)
#
# likelihood_ref = like_ref((xx, yy))
# likelihood = like((xx, yy))
# likelihood_ref *= np.max(likelihood_ref) ** -1
# likelihood *= np.max(likelihood) ** -1
#
# ratio = likelihood_ref/likelihood
#
# plt.imshow(likelihood_ref,vmin=0, vmax=1)
# plt.show()
# plt.imshow(likelihood, vmin=0, vmax=1)
# plt.show()
# plt.imshow(ratio, vmin=0, vmax=1)
# plt.show()
# plt.imshow(ratio * likelihood, vmin=0, vmax=1)
# plt.show()
