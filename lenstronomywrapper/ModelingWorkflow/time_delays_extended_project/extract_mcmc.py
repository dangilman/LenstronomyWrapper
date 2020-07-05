import dill as pickle
import os
import numpy as np
from lenstronomywrapper.Utilities.data_util import write_data_to_file
from lenstronomywrapper.Utilities.misc import create_directory
import sys
from lenstronomywrapper.ModelingWorkflow.time_delays_extended_project.MCMCchain import MCMCchainNew

def flux_ratio_anomaly(f1, f2):
    return f1 - f2


def time_delay_anomaly(t1, t2):
    return t1 - t2

def extract_observables(mcmc_chain_dir, output_dir, n, n_burn_frac, n_keep=200):

    fname = mcmc_chain_dir + '/MCMCchain_' + str(n)
    print(fname)
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

    fnames = ['tbaseline_', 'flux_anomaly_', 'time_anomaly_', 'time_delays_', 'macroparams_']

    logL = mcmc.modelPlot._imageModel.likelihood_data_given_model(
        source_marg=False, linear_prior=None, **mcmc.kwargs_result)
    ndata_points = mcmc.modelPlot._imageModel.num_data_evaluate
    chi2_imaging = logL * 2 / ndata_points

    arrays = [time_delay_baseline, flux_anomaly, time_anomaly, time_delays_model,
              kwargs_data_fit['kwargs_lens_macro_fit']]

    if not os.path.exists(output_dir):
        create_directory(output_dir)

    write_data_to_file(output_dir + 'chi2_imaging_' + str(n) + '.txt', chi2_imaging, 'w')
    for fname, arr in zip(fnames, arrays):
        write_data_to_file(output_dir + fname + str(n) + '.txt', arr, 'w')


def extract_maps(mcmc_chain_dir, output_dir, n):

    fname = mcmc_chain_dir + '/MCMCchain_' + str(n)
    print(fname)
    if os.path.exists(fname):
        file = open(fname, 'rb')
    else:
        return None

    mcmc = pickle.load(file)

    mcmc_new = MCMCchainNew(mcmc)

    kwargs_maps = mcmc_new.maps()

    # logL = mcmc.modelPlot._imageModel.likelihood_data_given_model(
    #     source_marg=False, linear_prior=None, **mcmc.kwargs_result)
    # ndata_points = mcmc.modelPlot._imageModel.num_data_evaluate
    # chi2_imaging = logL * 2 / ndata_points

    observed_lens = kwargs_maps['observed_lens']
    modeled_lens = kwargs_maps['modeled_lens']
    normalized_residuals = kwargs_maps['normalized_residuals']
    residual_convergence = kwargs_maps['residual_convergence']
    residual_mean_kappa = kwargs_maps['mean_kappa']
    time_delay_residuals = kwargs_maps['time_delay_residuals']
    reconstructed_source = kwargs_maps['reconstructed_source']

    if not os.path.exists(output_dir):
        create_directory(output_dir)

    np.savetxt(output_dir + 'observed_' + str(n) + '.txt', X=observed_lens)
    np.savetxt(output_dir + 'modeled_' + str(n) + '.txt', X=modeled_lens)
    np.savetxt(output_dir + 'residuals_' + str(n) + '.txt', X=normalized_residuals)
    np.savetxt(output_dir + 'kappa_' + str(n) + '.txt', X=residual_convergence)
    np.savetxt(output_dir + 'tdelayres_' + str(n) + '.txt', X=time_delay_residuals)
    np.savetxt(output_dir + 'source_' + str(n) + '.txt', X=reconstructed_source)

    write_data_to_file(output_dir + 'kappares_' + str(n) + '.txt', residual_mean_kappa, 'w')
    #write_data_to_file(output_dir + 'chi2_imaging_' + str(n) + '.txt', chi2_imaging, 'w')

try:
    base_path = os.getenv('HOME') + '/../../../../u/flashscratch/g/gilmanda'
    assert os.path.exists(base_path)
except:
    base_path = os.getenv('HOME') + '/Code'
    assert os.path.exists(base_path)

lensID = 'lens1115'
n_keep = 200
fname_base = base_path + '/tdelay_output/raw/' + lensID

extension = '/chains_' + 'shapelets_mlow6_fixDdt'
fname_chains = fname_base + extension
i_start = int(sys.argv[1])

burn_cut = 0.9
fname_out = fname_base + '/los_plus_subs_mlow6_shapelets_fixDdt/'
#extract_maps(fname, i_start, append)
extract_observables(fname_chains, fname_out, i_start, burn_cut, n_keep)

burn_cut = 0.75
fname_out = fname_base + '/los_plus_subs_mlow6_shapelets_fixDdt_convtest/'
#extract_maps(fname, i_start, append)
extract_observables(fname_chains, fname_out, i_start, burn_cut, n_keep)

