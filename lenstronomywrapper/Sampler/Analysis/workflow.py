import numpy as np
import matplotlib.pyplot as plt
import os
import dill
from MagniPy.Analysis.KDE.NDdensity import *
from MagniPy.Analysis.Visualization.triplot2 import TriPlot2

path_out = os.getenv('HOME') + '/data/sims/processed_chains/CDMforecast4/'

nbins = 10
n_keep = 600
samples_list, samples_list_weighted = [], []

param_names = ['sigma_sub', 'power_law_index', 'center_x', 'center_y']

for lens_idx in range(1, 2):

    file = open(path_out+'lens_'+str(lens_idx), 'rb')
    samples = dill.load(file)
    param_names_full = samples.param_name_list
    print(samples.param_name_list)
    print(samples.ranges_dictionary)

    sigma_scale = 1.
    flux_sigmas = [sigma_scale*0.01/0.88,
                   sigma_scale*0.01,
                   sigma_scale*0.006/0.47, None]

    iter = 1
    x, x_full, x_ranges, stats = samples.sample_with_flux_uncertainties(
        param_names, flux_sigmas, iter=iter, n_keep=n_keep, auto_range_sigma=3.5)
    print('kept '+str(len(x[:,0])/iter) + ' realizations...')
    print(np.max(stats), stats.shape)
    print(x.shape)
    print(x_ranges)

    # weight_names = ['center_x', 'center_y']
    # mean_list = [0., 0.]
    # sigma_list = [0.01, 0.01]
    #weights = [samples.gaussian_weight(x_full, weight_names,
    #                                  mean_list, sigma_list)]
    weights = None
    data = [x]

    samples_list.append(DensitySamples([x], param_names, weights,
                             param_ranges=x_ranges, nbins=nbins,
                             use_kde=False, bwidth_scale=0.4))

density = IndepdendentDensities(samples_list)
density_weighted = IndepdendentDensities(samples_list_weighted)
triplot = TriPlot2([density], param_names, x_ranges)
triplot.truth_color = 'b'
triplot.make_triplot(param_names=param_names,
                     filled_contours=False, truths=None, show_intervals=False,
                     show_contours=True, levels=[0.05, 0.32, 1])
#triplot.make_marginal(param_names[1])

#plt.savefig('example_inference.pdf')
plt.show()
