import dill as pickle
import os
import numpy as np
from lenstronomywrapper.Utilities.data_util import write_data_to_file

def flux_ratio_anomaly(f1, f2):
    return f1 - f2

def time_delay_anomaly(t1, t2):
    return t1 - t2

def extract_mcmc_chain_single(pickle_dir, idx, n_burn_frac, n_keep=100):

    fname = pickle_dir + '/MCMCchain_' + str(idx)

    file = open(fname, 'rb')
    mcmc = pickle.load(file)

    return_kwargs_fit, kwargs_data_fit, kwargs_data_setup = mcmc.get_output(n_burn_frac, n_keep)

    save_name_path = mcmc.save_name_path

    macromodel_params = np.round(return_kwargs_fit['kwargs_lens_macro_fit'], 5)
    srcx, srcy = np.round(kwargs_data_fit['source_x'], 4), np.round(kwargs_data_fit['source_y'], 4)

    macromodel_params = np.hstack((macromodel_params, np.array([srcx, srcy]).reshape(len(srcx), 2)))

    key = 'flux_ratios'
    flux_anomaly = np.round(flux_ratio_anomaly(kwargs_data_fit[key], kwargs_data_setup[key]), 4)

    key = 'time_delays'
    time_anomaly = np.round(time_delay_anomaly(kwargs_data_fit[key], kwargs_data_setup[key]), 4)
    time_delays_model = kwargs_data_fit[key]
    time_delay_baseline = kwargs_data_setup[key]

    return time_delay_baseline, flux_anomaly, time_anomaly, time_delays_model, \
                macromodel_params, return_kwargs_fit, kwargs_data_setup, save_name_path

def extract_mcmc_chains(pickle_dir, idx_min, idx_max, n_burn_frac, n_keep=200, save_name_path_append=''):

    for n in range(idx_min, idx_max):

        tbaseline, f, t, tdelay_model, macro_params, kw_fit, kw_setup, save_name_path = extract_mcmc_chain_single(
            pickle_dir, n, n_burn_frac, n_keep
        )

        observed_lens = kw_fit['observed_lens']
        modeled_lens = kw_fit['modeled_lens']
        normalized_residuals = kw_fit['normalized_residuals']
        residual_convergence = kw_fit['residual_convergence']
        residual_mean_kappa = kw_fit['mean_kappa']
        time_delay_residuals = kw_fit['time_delay_residuals']
        reconstructed_source = kw_fit['reconstructed_source']

        fnames = ['tbaseline_', 'flux_anomaly_', 'time_anomaly_', 'time_delays_', 'macroparams_',
              'kappares_', 'chi2_imaging_']

        arrays = [tbaseline, f, t, tdelay_model,
              macro_params, residual_mean_kappa, kw_fit['chi2_imaging']]

        for fname, arr in zip(fnames, arrays):
            write_data_to_file(save_name_path + fname + str(n) + '.txt', arr, 'w')

        save_name_path += save_name_path_append
        print(save_name_path)
        np.savetxt(save_name_path + 'observed_' + str(n) + '.txt', X=observed_lens)
        np.savetxt(save_name_path + 'modeled_' + str(n) + '.txt', X=modeled_lens)
        np.savetxt(save_name_path + 'residuals_' + str(n) + '.txt', X=normalized_residuals)
        np.savetxt(save_name_path + 'kappa_' + str(n) + '.txt', X=residual_convergence)
        np.savetxt(save_name_path + 'tdelayres_' + str(n) + '.txt', X=time_delay_residuals)
        np.savetxt(save_name_path + 'source_' + str(n) + '.txt', X=reconstructed_source)

    return None

try:
    base_path = os.getenv('HOME') + '/../../../../u/flashscratch/g/gilmanda'
    assert os.path.exists(base_path)
except:
    base_path = os.getenv('HOME') + '/Code'
    assert os.path.exists(base_path)

idx = 1
lensID = 'lens0435'

extension = 'mlow8.7test_fixDdt'
fname = base_path + '/tdelay_output/raw/' + lensID + '/realizations_' + extension

extract_mcmc_chains(fname, 1, 4, 0.5)
