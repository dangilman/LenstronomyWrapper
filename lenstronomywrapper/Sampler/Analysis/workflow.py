import numpy as np
import matplotlib.pyplot as plt
import os
import dill
from MagniPy.Analysis.KDE.NDdensity import *
from MagniPy.Analysis.Visualization.triplot2 import TriPlot2

paths = [os.getenv('HOME') + '/data/sims/processed_chains/CDMforecast4/',
         os.getenv('HOME') + '/data/sims/processed_chains/HE0435/',
         os.getenv('HOME') + '/data/sims/processed_chains/WGD2038/']

nbins = 15
n_keep = 600

sigmas_1422 = [0.01/0.88, 0.01, 0.006/0.47, None]
sigmas_0435 = [0.05, 0.049, 0.048, 0.056]
sigmas_2038 = [0.01, 0.017, 0.022, 0.022]
flux_sigmas_list = [sigmas_1422, sigmas_0435, sigmas_2038]
flux_sigmas_list = [sigmas_0435]
paths = [paths[1]]
param_names = ['sigma_sub', 'power_law_index']
samples_list = []
samples_list_weighted = []

for n, path_out in enumerate(paths):
    if n>3:
        continue
    print(path_out + 'lens_'+str(1))
    file = open(path_out+'lens_'+str(1), 'rb')
    samples = dill.load(file)
    param_names_full = samples.param_name_list
    print(samples.param_name_list)
    print(samples.ranges_dictionary)

    iter = 5
    x, x_full, x_ranges, stats = samples.sample_with_flux_uncertainties(
        param_names, flux_sigmas_list[n], iter=iter, n_keep=n_keep, auto_range_sigma=3.5)
    print('kept '+str(len(x[:,0])/iter) + ' realizations...')
    print(np.max(stats), stats.shape)
    print(x.shape)
    print(x_ranges)

    weight_names = ['center_x', 'center_y']
    mean_list = [0., 0.]
    sigma_list = [0.05, 0.05]
    weights = [samples.gaussian_weight(x_full, weight_names,
                                       mean_list, sigma_list)]

    samples_list_weighted.append(DensitySamples([x], param_names, weights,
                                       param_ranges=x_ranges, nbins=nbins,
                                       use_kde=True, bwidth_scale=0.3))

sim_weighted = IndepdendentDensities(samples_list_weighted)

levels = [0.05, 0.32, 1.]
triplot = TriPlot2([sim_weighted], param_names, x_ranges)
triplot.truth_color = 'b'
out = triplot.make_triplot(param_names=param_names,
                     filled_contours=False, truths=None, show_intervals=False,
                     show_contours=True, levels=levels)
out[1].annotate('HE0435', xy=(0.5, 0.5), xycoords='axes fraction', fontsize=16)
#triplot.make_marginal(param_names[1])

#plt.savefig('example_inference.pdf')
plt.show()
