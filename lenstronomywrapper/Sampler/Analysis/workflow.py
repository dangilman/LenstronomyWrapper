import numpy as np
import matplotlib.pyplot as plt
import os
import dill
from MagniPy.Analysis.KDE.NDdensity import *
from MagniPy.Analysis.Visualization.triplot2 import TriPlot2

sigmas_1422 = ([0.01/0.88, 0.01, 0.006/0.47, None], False)
sigmas_0435 = ([0.05, 0.049, 0.048, 0.056], False)
sigmas_2038 = ([0.01, 0.017, 0.022, 0.022], False)
sigmas_1606 = ([0.03, 0.03, 0.02/0.59, 0.02/0.79], False)
sigmas_1115 = ([0.01] * 3, True)
sigmas_0405 = ([0.04, 0.03/0.7, 0.04/1.28, 0.05/0.94], False)
sigmas_2033 = ([0.03, 0.03/0.65, 0.02/0.5, 0.02/0.53], False)

base = os.getenv('HOME') + '/data/sims/processed_chains/'
names = ['B1422_varyLOSindex', 'PG1115', 'HE0435', 'PSJ1606', 'WGD2038',
         'WGDJ0405', 'WFI2033']
paths_base = {name: base + name for name in names}

sigmas_dict = {'B1422_varyLOSindex': sigmas_1422, 'PG1115': sigmas_1115,
               'HE0435': sigmas_0435, 'PSJ1606': sigmas_1606,
              'WGD2038': sigmas_2038, 'WGDJ0405': sigmas_0405,
              'WFI2033': sigmas_2033}

nbins = 15
n_keep = 400

param_names = ['sigma_sub', 'power_law_index']
samples_list_weighted = []

include_list = ['B1422_varyLOSindex', 'PG1115', 'PSJ1606', 'WGDJ0405',
                'WFI2033', 'HE0435', 'WGD2038']
include_list = ['HE0435']
paths = [paths_base[lens_name] + '/' for lens_name in include_list]
flux_sigmas_list = [sigmas_dict[lens_name][0] for lens_name in include_list]
uncertainty_in_ratios = [sigmas_dict[lens_name][1] for lens_name in include_list]

for n, path_out in enumerate(paths):

    print(path_out + 'lens_'+str(1))
    file = open(path_out+'lens_'+str(1), 'rb')
    samples = dill.load(file)
    param_names_full = samples.param_name_list

    iter = 5

    x, x_full, x_ranges, stats = samples.sample_with_flux_uncertainties(
        param_names, flux_sigmas_list[n], uncertainty_in_ratios[n], iter=iter, n_keep=n_keep, auto_range_sigma=3.5)
    print('kept '+str(len(x[:,0])/iter) + ' realizations...')
    print(np.max(stats), stats.shape)
    print(x.shape)
    print(x_ranges)
    #x_ranges = [[0, 0.2], [-2.7, -1.1], [-0.03, 0.17]]

    weight_names = ['center_x', 'center_y', 'LOS_normalization']
    mean_list = [0., 0., 0.8]
    sigma_list = [0.05, 0.05, 1.]
    weights = [samples.gaussian_weight(x_full, weight_names,
                                       mean_list, sigma_list)]

    samples_list_weighted.append(DensitySamples([x], param_names, weights,
                                       param_ranges=x_ranges, nbins=nbins,
                                       use_kde=True, bwidth_scale=0.4))

sim_weighted = IndepdendentDensities(samples_list_weighted)

levels = [0.05, 0.32, 1.]
triplot = TriPlot2([sim_weighted], param_names, x_ranges)
triplot.truth_color = 'b'
out = triplot.make_triplot(param_names=param_names,
                     filled_contours=True, show_intervals=False,
                     show_contours=True, levels=levels)
#out[1].annotate('4 lenses', xy=(0.5, 0.5), xycoords='axes fraction', fontsize=16)
#triplot.make_marginal(param_names[1])

plt.savefig('posterior_vary_SHMF.pdf')
plt.show()
