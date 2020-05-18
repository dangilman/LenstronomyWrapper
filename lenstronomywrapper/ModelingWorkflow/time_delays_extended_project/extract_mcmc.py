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

def extract_mcmc_chains(pickle_dir, idx_min, idx_max, n_burn_frac, n_keep=200):

    observed_lens, modeled_lens, normalized_residuals, residual_convergence = [], [], [], []
    residual_mean_kappa = []
    time_delay_residuals = []
    reconstructed_source = []

    init = True

    for n in range(idx_min, idx_max):

        tbaseline, f, t, tdelay_model, macro_params, kw_fit, kw_setup, save_name_path = extract_mcmc_chain_single(
            pickle_dir, n, n_burn_frac, n_keep
        )

        observed_lens.append(kw_fit['observed_lens'])
        modeled_lens.append(kw_fit['modeled_lens'])
        normalized_residuals.append(kw_fit['normalized_residuals'])
        residual_convergence.append(kw_fit['residual_convergence'])
        residual_mean_kappa.append(kw_fit['mean_kappa'])
        time_delay_residuals.append(kw_fit['time_delay_residuals'])
        reconstructed_source.append(kw_fit['reconstructed_source'])

        if init:
            baseline = tbaseline
            flux_anomalies = f
            time_anomalies = t
            chi2_imaging = kw_fit['chi2_imaging'].ravel()
            time_delays_model = tdelay_model
            # ddt_inferred = kw_fit['D_dt_samples'].ravel()
            macromodel_parameters = macro_params
            init = False

        else:
            baseline = np.vstack((baseline, tbaseline))
            flux_anomalies = np.vstack((flux_anomalies, f))
            time_anomalies = np.vstack((time_anomalies, t))
            time_delays_model = np.vstack((time_delays_model, tdelay_model))
            chi2_imaging = np.append(chi2_imaging, kw_fit['chi2_imaging'].ravel())
            # ddt_inferred = np.append(ddt_inferred, kw_fit['D_dt_samples'].ravel())
            # h0_inferred = np.append(h0_inferred, h0_inf.ravel()).flatten()
            macromodel_parameters = np.vstack((macromodel_parameters, macro_params))

    fnames = ['tbaseline_', 'flux_anomaly_', 'time_anomaly_', 'time_delays_', 'macroparams_',
              'time_delay_sigma_', 'kappares_', 'chi2_imaging_']

    arrays = [baseline, flux_anomalies, time_anomalies, time_delays_model,
              macromodel_parameters, np.array(residual_mean_kappa), chi2_imaging]

    for fname, arr in zip(fnames, arrays):
        write_data_to_file(save_name_path + fname + str(n) + '.txt', arr)

    for i in range(0, len(observed_lens)):
        print(save_name_path)
        np.savetxt(save_name_path + 'observed_' + str(n+1) + '.txt', X=observed_lens[i])
        np.savetxt(save_name_path + 'modeled_' + str(n+1) + '.txt', X=modeled_lens[i])
        np.savetxt(save_name_path + 'residuals_' + str(n+1) + '.txt', X=normalized_residuals[i])
        np.savetxt(save_name_path + 'kappa_' + str(n+1) + '.txt', X=residual_convergence[i])
        np.savetxt(save_name_path + 'tdelayres_' + str(n+1) + '.txt', X=time_delay_residuals[i])
        np.savetxt(save_name_path + 'source_' + str(n+1) + '.txt', X=reconstructed_source[i])

    return [flux_anomalies, baseline, time_anomalies]

try:
    base_path = os.getenv('HOME') + '/../../../../u/flashscratch/g/gilmanda'
    assert os.path.exists(base_path)
except:
    base_path = os.getenv('HOME') + '/Code'
    assert os.path.exists(base_path)

idx = 1
lensID = 'lens1115'
base_path = os.getenv('HOME') + '/Code'
extension = 'mlow6_fixDdt'
fname = base_path + '/tdelay_output/raw/' + lensID + '/realizations_' + extension

extract_mcmc_chains(fname, 1, 2, 0.5)
