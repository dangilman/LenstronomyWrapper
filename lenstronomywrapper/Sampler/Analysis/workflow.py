import numpy as np
import matplotlib.pyplot as plt
import os
import dill

path_out = os.getenv('HOME') + '/data/sims/processed_chains/benson_run_1/'

file = open(path_out+'lens_1', 'rb')
samples_1 = dill.load(file)

new_params = [0.1, 0.1, 2.05, 1.]
sigmas = [0.001, 0.1, 1, 0.25]

samples, stats = samples_1.sample_with_flux_uncertainties(0, 300)
weights = np.exp(-0.5 * (samples[:,1] - 0.01)**2/1.**2)
weights *= np.exp(-0.5 * (samples[:,2] - 2.05)**2/0.025**2)
#samples, stats = samples_new.sample_with_flux_uncertainties(0, 200)
plt.hist(samples[:,0], weights=weights)
plt.show()
