import numpy as np
import matplotlib.pyplot as plt
import os
import dill
from MagniPy.Analysis.KDE.NDdensity import *
from MagniPy.Analysis.Visualization.triplot2 import TriPlot2

paths = [os.getenv('HOME') + '/data/sims/processed_chains/CDMforecast4/',
         os.getenv('HOME') + '/data/sims/processed_chains/PG1115/',
         os.getenv('HOME') + '/data/sims/processed_chains/HE0435/',
         os.getenv('HOME') + '/data/sims/processed_chains/PSJ1606/']

nbins = 6
n_keep = 400
sigmas_1422 = [0.01/0.88, 0.01, 0.006/0.47, None]
sigmas_0435 = [0.05, 0.049, 0.048, 0.056]
sigmas_2038 = [0.01, 0.017, 0.022, 0.022]
sigmas_1606 = [0.03, 0.03, 0.02/0.59, 0.02/0.79]
sigmas_1115 = [0.01] * 4
flux_sigmas_list = [sigmas_1422, sigmas_1115, sigmas_0435, sigmas_1606]
#flux_sigmas_list = [sigmas_1606]

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

    iter = 2
    #new_param_truths = {'sigma_sub': 0.13, 'power_law_index': -1.9}
    #new_param_sigmas = {'sigma_sub': 0.01, 'power_law_index': 0.05}
    #samples = samples.resample_new_parameters(new_param_truths, new_param_sigmas)
    x, x_full, x_ranges, stats = samples.sample_with_flux_uncertainties(
        param_names, flux_sigmas_list[n], iter=iter, n_keep=n_keep, auto_range_sigma=3.5)
    print('kept '+str(len(x[:,0])/iter) + ' realizations...')
    print(np.max(stats), stats.shape)
    print(x.shape)
    print(x_ranges)
    x_ranges = [[0, 0.2], [-2.7, -1.2]]

    weight_names = ['center_x', 'center_y']
    mean_list = [0., 0., 0.8]
    sigma_list = [0.05, 0.05]
    weights = [samples.gaussian_weight(x_full, weight_names,
                                       mean_list, sigma_list)]

    samples_list_weighted.append(DensitySamples([x], param_names, weights,
                                       param_ranges=x_ranges, nbins=nbins,
                                       use_kde=False, bwidth_scale=0.3))

sim_weighted = IndepdendentDensities(samples_list_weighted)

levels = [0.05, 0.32, 1.]
triplot = TriPlot2([sim_weighted], param_names, x_ranges)
triplot.truth_color = 'b'
out = triplot.make_triplot(param_names=param_names,
                     filled_contours=False, truths=None, show_intervals=False,
                     show_contours=True, levels=levels)
#out[1].annotate('4 lenses', xy=(0.5, 0.5), xycoords='axes fraction', fontsize=16)
#triplot.make_marginal(param_names[1])

plt.savefig('posterior.pdf')
plt.show()
