import dill as pickle
import os
import numpy as np
from lenstronomywrapper.Utilities.data_util import write_data_to_file
import sys
from lenstronomywrapper.ModelingWorkflow.time_delays_extended_project.MCMCchain import MCMCchainNew

def flux_ratio_anomaly(f1, f2):
    return f1 - f2


def time_delay_anomaly(t1, t2):
    return t1 - t2


def extract_mcmc_chain_single(pickle_dir, idx, n_burn_frac, n_keep=100):
    fname = pickle_dir + '/MCMCchain_' + str(idx)

    if os.path.exists(fname):
        file = open(fname, 'rb')
    else:
        return None, None, None, None, None, None, None, None

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

def extract_observables(pickle_dir, n, n_burn_frac, n_keep=200, save_name_path_append=None):

    fname = pickle_dir + '/MCMCchain_' + str(idx)

    if os.path.exists(fname):
        file = open(fname, 'rb')
    else:
        return None

    mcmc = pickle.load(file)

    mcmc_new = MCMCchainNew(mcmc)

    kwargs_data_fit, kwargs_data_setup = mcmc_new.extract_observables(n_burn_frac, n_keep)

    key = 'flux_ratios'
    flux_anomaly = np.round(flux_ratio_anomaly(kwargs_data_fit[key], kwargs_data_setup[key]), 4)

    key = 'time_delays'
    time_anomaly = np.round(time_delay_anomaly(kwargs_data_fit[key], kwargs_data_setup[key]), 4)
    time_delays_model = kwargs_data_fit[key]
    time_delay_baseline = kwargs_data_setup[key]

    if save_name_path_append is not None:
        outpath = mcmc.save_name_path[0:-1] + save_name_path_append + '/'

    else:
        outpath = mcmc.save_name_path

    fnames = ['tbaseline_', 'flux_anomaly_', 'time_anomaly_', 'time_delays_', 'macroparams_']

    arrays = [time_delay_baseline, flux_anomaly, time_anomaly, time_delays_model,
              kwargs_data_fit['kwargs_lens_macro_fit']]

    for fname, arr in zip(fnames, arrays):
        write_data_to_file(outpath + fname + str(n) + '.txt', arr, 'w')


def extract_maps(pickle_dir, n, save_name_path_append=None):

    fname = pickle_dir + '/MCMCchain_' + str(idx)
    print(fname)
    if os.path.exists(fname):
        file = open(fname, 'rb')
    else:
        return None

    mcmc = pickle.load(file)

    mcmc_new = MCMCchainNew(mcmc)

    kwargs_maps = mcmc_new.maps()

    logL = mcmc.modelPlot._imageModel.likelihood_data_given_model(
        source_marg=False, linear_prior=None, **mcmc.kwargs_result)
    ndata_points = mcmc.modelPlot._imageModel.num_data_evaluate
    chi2_imaging = logL * 2 / ndata_points

    observed_lens = kwargs_maps['observed_lens']
    modeled_lens = kwargs_maps['modeled_lens']
    normalized_residuals = kwargs_maps['normalized_residuals']
    residual_convergence = kwargs_maps['residual_convergence']
    residual_mean_kappa = kwargs_maps['mean_kappa']
    time_delay_residuals = kwargs_maps['time_delay_residuals']
    reconstructed_source = kwargs_maps['reconstructed_source']

    if save_name_path_append is not None:
        outpath = mcmc.save_name_path[0:-1] + save_name_path_append + '/'

    else:
        outpath = mcmc.save_name_path
    print(outpath)
    a=input('continue')
    np.savetxt(outpath + 'observed_' + str(n) + '.txt', X=observed_lens)
    np.savetxt(outpath + 'modeled_' + str(n) + '.txt', X=modeled_lens)
    np.savetxt(outpath + 'residuals_' + str(n) + '.txt', X=normalized_residuals)
    np.savetxt(outpath + 'kappa_' + str(n) + '.txt', X=residual_convergence)
    np.savetxt(outpath + 'tdelayres_' + str(n) + '.txt', X=time_delay_residuals)
    np.savetxt(outpath + 'source_' + str(n) + '.txt', X=reconstructed_source)

    write_data_to_file(outpath + 'kappares_' + str(n) + '.txt', residual_mean_kappa, 'w')
    write_data_to_file(outpath + 'chi2_imaging_' + str(n) + '.txt', chi2_imaging, 'w')

def extract_mcmc_chains(pickle_dir, n, n_burn_frac, n_keep=200, save_name_path_append=None):

    tbaseline, f, t, tdelay_model, macro_params, kw_fit, kw_setup, save_name_path = extract_mcmc_chain_single(
        pickle_dir, n, n_burn_frac, n_keep
    )
    if tbaseline is None:
        return

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

    if save_name_path_append is not None:
        outpath = save_name_path[0:-1]+ save_name_path_append +'/'

    else:
        outpath = save_name_path

    for fname, arr in zip(fnames, arrays):
        write_data_to_file(outpath + fname + str(n) + '.txt', arr, 'w')

    np.savetxt(outpath + 'observed_' + str(n) + '.txt', X=observed_lens)
    np.savetxt(outpath + 'modeled_' + str(n) + '.txt', X=modeled_lens)
    np.savetxt(outpath + 'residuals_' + str(n) + '.txt', X=normalized_residuals)
    np.savetxt(outpath + 'kappa_' + str(n) + '.txt', X=residual_convergence)
    np.savetxt(outpath + 'tdelayres_' + str(n) + '.txt', X=time_delay_residuals)
    np.savetxt(outpath + 'source_' + str(n) + '.txt', X=reconstructed_source)

    return

try:
    base_path = os.getenv('HOME') + '/../../../../u/flashscratch/g/gilmanda'
    assert os.path.exists(base_path)
except:
    base_path = os.getenv('HOME') + '/Code'
    assert os.path.exists(base_path)

lensID = 'lens1115'
n_keep = 200
extension = 'control_shapelets_mlow6_fixDdt'
fname = base_path + '/tdelay_output/raw/' + lensID + '/chains_' + extension
i_start = int(sys.argv[1])

append = None
burn_cut = 0.9
#extract_maps(fname, i_start, append)
extract_observables(fname, i_start, burn_cut, n_keep, append)

append = '_convtest'
burn_cut = 0.75
#extract_maps(fname, i_start, append)
extract_observables(fname, i_start, burn_cut, n_keep, append)

