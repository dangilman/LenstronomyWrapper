import numpy as np
import matplotlib.pyplot as plt
import os
import dill
from MagniPy.Analysis.KDE.NDdensity import *
from MagniPy.Analysis.Visualization.triplot2 import TriPlot2

path_out = os.getenv('HOME') + '/data/sims/processed_chains/coldSIDM/'

lens_idx = 10
nbins = 10
n_keep = 200
samples_list, samples_list_weighted = [], []

for lens_idx in range(1, 16):
    file = open(path_out+'lens_'+str(lens_idx), 'rb')
    samples = dill.load(file)
    param_names = samples.param_name_list
    print(samples.param_name_list)
    new_param_truths = [0.06, 5.7, 2.08, 1.]
    new_param_sigmas = [0.005, 0.1, 2., 5.]
    samples = samples.resample_new_parameters(new_param_truths, new_param_sigmas)

    x, stats = samples.sample_with_flux_uncertainties(0, n_keep, discard=0)

    #ind = np.where(stats > 0.1)[0]

    #x = np.delete(x, 2, axis=1)

    #truths = {param_names[0]: new_param_truths[0], param_names[1]: new_param_truths[1],
    #          param_names[3]: new_param_truths[3]}
    truths = None
    param_names = [param_names[0], param_names[1]]

    if lens_idx == 1:
        dx1 = (x[:, 0] - new_param_truths[0]) / 10
        dx3 = (x[:, 2] - new_param_truths[2])/ 10.5
        weights = [np.exp(-0.5 * (dx1 ** 2 + dx3**2))]
    else:
        weights = None
    weights = None
    x = np.delete(x, (2,3), axis=1)
    data = [x]

    #param_ranges = [[0, 0.1], [0.01, 10], [2, 2.1], [0.25, 5]]
    param_ranges = [[0, 0.1], [0.01, 10]]

    samples_list.append(DensitySamples([x], param_names, None,
                             param_ranges=param_ranges, nbins=nbins,
                             use_kde=True))
    samples_list_weighted.append(DensitySamples([x], param_names, weights,
                             param_ranges=param_ranges, nbins=nbins,
                             use_kde=True))

density = IndepdendentDensities(samples_list)
density_weighted = IndepdendentDensities(samples_list_weighted)
triplot = TriPlot2([density], param_names, param_ranges)
triplot.truth_color = 'b'
triplot.make_triplot(param_names=param_names,
                     filled_contours=False, truths=truths, show_intervals=False)
#triplot.make_marginal(param_names[1])

#plt.savefig('example_inference.pdf')
plt.show()
