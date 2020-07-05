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

def transform_pdf_from_samples_2d(density_samples, density_samples_ref,
                                  pname1, pname2, pranges, nbins):


    d_control = density_samples_ref.projection_2D(pname1, pname2)
    d = density_samples.projection_2D(pname1, pname2)
    d_control *= np.max(d_control) ** -1
    d *= np.max(d) ** -1
    re_weight = d_control / d
    nan_inds = np.isnan(re_weight)
    re_weight[nan_inds] = 0
    xran, yran = pranges[0], pranges[1]
    points = (np.linspace(*xran, nbins), np.linspace(*yran, nbins))
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(points, re_weight.T)

    def _func(xpoints, ypoints):

        out = []
        for i, (xi, yi) in enumerate(zip(xpoints, ypoints)):

            try:
                value = interp((xi, yi))
                out.append(value)
            except:

                out.append(0)
        return np.squeeze(out)

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
                          fill_value=0., kind='nearest')

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

# cov1 = [[0.05, -0.035], [-0.035, 0.05]]
# cov2 = [[0.1, -0.0], [-0.0, 0.1]]
#
# samplesref = np.random.multivariate_normal([0, 0.], cov1, 50000)
# samples = np.random.multivariate_normal([0, 0], cov2, 50000)
#
# import matplotlib.pyplot as plt
#
# pran1, pran2 = [-1, 1], [-1, 1]
# func = transform_pdf_from_samples_2d([samples[:,0], samples[:,1]],
#                                      [samplesref[:,0], samplesref[:,1]],
#                                      100, pran1, pran2)
#
# href, _, _ = np.histogram2d(samplesref[:,0], samplesref[:,1], bins=20, density=True,
#                             range=(pran1, pran2))
# h, xran, yran = np.histogram2d(samples[:,0], samples[:,1],
#                          bins=20, density=True, range=(pran1, pran2))
# h *= np.max(h) ** -1
# href *= np.max(href) ** -1
#
# step1, step2 = (xran[1] - xran[0])/2, (yran[1] - yran[0])/2
# xran, yran = xran[0:-1] + step1, yran[0:-1] + step2
# xx, yy = np.meshgrid(xran, yran)
# xx, yy = xx.ravel(), yy.ravel()
#
# weights = func(samples[:,0], samples[:,1])
#
# h_weighted, _, _ = np.histogram2d(samples[:,0], samples[:,1],
#                          bins=20, density=True, weights=weights,
#                                   range=(pran1, pran2))
# h_weighted *= np.max(h_weighted) ** -1
#
# plt.imshow(href,vmin=0, vmax=1)
# plt.show()
# plt.imshow(h, vmin=0, vmax=1)
# plt.show()
# plt.imshow(h_weighted * h / np.max(h_weighted * h), vmin=0, vmax=1)
# plt.show()
# plt.imshow(h_weighted, vmin=0, vmax=1)
# plt.show()
