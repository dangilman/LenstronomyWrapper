import numpy as np
from lenstronomywrapper.Sampler.Analysis.single_lens_samples import LensSamplesRaw
import matplotlib.pyplot as plt
import os
import dill
from MagniPy.Analysis.KDE.NDdensity import *
from MagniPy.Analysis.Visualization.triplot2 import TriPlot2

from MagniPy.Workflow.mocks.B1422mock import B1422Mock
from MagniPy.Workflow.mocks.HE0435mock import Lens0435Mock
from MagniPy.Workflow.mocks.PSJ1606mock import Lens1606Mock
from MagniPy.Workflow.mocks.WFI2033mock import WFI2033Mock
from MagniPy.Workflow.mocks.WGD2038Mock import Lens2038Mock
from MagniPy.Workflow.mocks.RXJ0911mock import Lens0911Mock

sigmas_1422 = ([0.01/0.88, 0.01, 0.006/0.47, None], False)
sigmas_0435 = ([0.05, 0.049, 0.048, 0.056], False)
sigmas_2038 = ([0.01, 0.017, 0.022, 0.022], False)
sigmas_1606 = ([0.03, 0.03, 0.02/0.59, 0.02/0.79], False)
sigmas_1115 = ([0.01] * 3, True)
sigmas_0405 = ([0.04, 0.03/0.7, 0.04/1.28, 0.05/0.94], False)
sigmas_2033 = ([0.03, 0.03/0.65, 0.02/0.5, 0.02/0.53], False)
sigmas_0911 = ([0.04/0.56, 0.05, 0.04/0.53, 0.04/0.24], False)
sigma_scale = 1.

base = os.getenv('HOME') + '/data/sims/processed_chains/'

names_base = ['B1422', 'PG1115', 'HE0435', 'PSJ1606', 'WGD2038',
         'WGDJ0405', 'WFI2033', 'RXJ0911']

sigmas_dict = {'B1422': sigmas_1422, 'PG1115': sigmas_1115,
               'HE0435': sigmas_0435, 'PSJ1606': sigmas_1606,
              'WGD2038': sigmas_2038, 'WGDJ0405': sigmas_0405,
              'WFI2033': sigmas_2033, 'RXJ0911': sigmas_0911}
mock_lenses = {'B1422': B1422Mock(), 'HE0435': Lens0435Mock(),
               'PSJ1606': Lens1606Mock(), 'WFI2033': WFI2033Mock(),
               'WGD2038': Lens2038Mock(), 'RXJ0911': Lens0911Mock()}

nbins = 12
n_keep = 600

param_names = ['sigma_sub', 'delta_power_law_index', 'c0', 'beta', 'LOS_normalization']
param_names = ['center_x', 'center_y', 'ellip_PA']
samples_list_weighted = []
samples_list_weighted_2 = []

include_list_master = ['WFI2033', 'HE0435', 'B1422', 'PSJ1606',
                'WGD2038', 'WGDJ0405', 'RXJ0911']
extension = ''
include_list_master = ['RXJ0911']
for lens_name in include_list_master:
    include_list = [lens_name]

    names = [nb + extension for nb in names_base]
    paths_base = {name: base + name for name in names}

    paths = [paths_base[lens_name + extension] + '/' for lens_name in include_list]
    flux_sigmas_list = []
    for lens_name in include_list:
        new = []
        for val in sigmas_dict[lens_name][0]:
            if val is not None:
                new.append(val * sigma_scale)
            else:
                new.append(val)
        flux_sigmas_list.append(new)

    uncertainty_in_ratios = [sigmas_dict[lens_name][1] for lens_name in include_list]

    for n, path_out in enumerate(paths):
        print(path_out)

        print(path_out + 'lens_'+str(1))
        file = open(path_out+'lens_'+str(1), 'rb')
        samples = dill.load(file)
        param_names_full = samples.param_name_list
        print(samples.ranges_dictionary)
        iter = 1
        #samples = samples.apply_param_cuts(samples, {'sigma_sub': [0., 0.1]})
        x, x_full, x_ranges, stats = samples.sample_with_flux_uncertainties(
            param_names, flux_sigmas_list[n], uncertainty_in_ratios[n], iter=iter, n_keep=n_keep, auto_range_sigma=3.5)

        print('kept '+str(len(x[:,0])/iter) + ' realizations...')
        print(np.max(stats), stats.shape)
        print(x.shape)
        print(x_ranges)

        #x_ranges = [[-0.5, 0.8], [0, 1.25], [0., 2], [0, 2]]

        weight_names_hyper = ['power_law_index']
        weight_names = ['center_x', 'center_y']

        mean_list_hyper = [-1.9]
        sigma_list_hyper = [0.05]

        mean_list = [0.0, 0.0]
        sigma_list = [100, 100]

        apply_hyper = [0]

        if n in apply_hyper:
            weights = [samples.gaussian_weight(
                x_full, weight_names_hyper + weight_names,
                mean_list_hyper + mean_list, sigma_list_hyper + sigma_list)]

        else:
            weights = [samples.gaussian_weight(
                x_full, weight_names, mean_list, sigma_list)]

        samples_list_weighted.append(DensitySamples([x], param_names, weights,
                                           param_ranges=x_ranges, nbins=nbins,
                                           use_kde=True, bwidth_scale=0.3))

    sim_weighted = IndepdendentDensities(samples_list_weighted)

    levels = [0.05, 0.32, 1.]
    print(samples.param_name_list)
    triplot = TriPlot2([sim_weighted], param_names, x_ranges)
    triplot.truth_color = 'b'
    triplot.get_parameter_confidence_interval(param_names[0], 2)
    triplot.get_parameter_confidence_interval(param_names[1], 2)
    triplot.get_parameter_confidence_interval('ellip_PA', 2)
    out = triplot.make_triplot(param_names=param_names, truths=None,
                         filled_contours=False, show_intervals=False,
                         show_contours=True, levels=levels)

    #plt.savefig('mock_8.pdf')

    plt.show()
    a=input('continue')
