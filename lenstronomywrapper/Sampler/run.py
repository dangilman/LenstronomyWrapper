import numpy as np
import os

from lenstronomywrapper.Sampler.utilities import *
from lenstronomywrapper.Utilities.misc import create_directory
from lenstronomywrapper.LensSystem.quad_lens import QuadLensSystem
from lenstronomywrapper.Utilities.data_util import approx_theta_E

from lenstronomywrapper.Utilities.parameter_util import kwargs_e1e2_to_polar, kwargs_gamma1gamma2_to_polar

from pyHalo.pyhalo_dynamic import pyHaloDynamic

from lenstronomywrapper.Optimization.quad_optimization.dynamic import DynamicOptimization

def run(job_index, output_path, path_to_folder):

    if not os.path.exists(output_path):
        print('creating directory '+ output_path)
        create_directory(output_path)

    if not os.path.exists(output_path):
        create_directory(output_path)

    keyword_arguments = load_keywords(path_to_folder, job_index)
    Nsamples = keyword_arguments['Nsamples']

    readout_steps = keyword_arguments['readout_steps']
    N_computed = 0
    readout_path = output_path + 'chain_' + str(job_index) + '/'

    if not os.path.exists(readout_path):
        create_directory(readout_path)

    fname_fluxes = readout_path + 'fluxes.txt'
    fname_params = readout_path + 'parameters.txt'
    fluxes_computed, parameters_sampled = None, None
    write_header = True

    if os.path.exists(fname_fluxes):
        write_header = False
        fluxes_computed = np.loadtxt(fname_fluxes)
        parameters_sampled = np.loadtxt(fname_params, skiprows=1)
        N_computed = int(fluxes_computed.shape[0])

    n_run = Nsamples - N_computed

    if n_run == 0:
        print('job index '+str(job_index) + ' finished.')
        return
    else:
        print('running job index '+str(job_index) + ', '+str(n_run) + ' realizations remaining')

    prior_list_realization, \
    prior_list_macromodel, \
    prior_list_source, \
    prior_list_cosmo = \
        build_priors(keyword_arguments['params_to_vary'])

    data_to_fit_init = load_data_to_fit(keyword_arguments)

    theta_E_approx = approx_theta_E(data_to_fit_init.x, data_to_fit_init.y)

    ############################ EVERYTHING BELOW THIS IS SAMPLED IN A FOR LOOP ############################

    pyhalo_dynamic = None
    kwargs_macro = []
    initialize = True
    kwargs_macro_ref = None

    for i in range(0, n_run):

        params_sampled = {}
        parameters = []

        ######## Sample keyword arguments for the substructure realization ##########

        kwargs_rendering, realization_samples = realization_keywords(keyword_arguments, prior_list_realization)
        params_sampled.update(realization_samples)

        ######## Sample keyword arguments for the lensing volume ##########
        zlens, zsource, lens_source_sampled = load_lens_source(prior_list_cosmo, keyword_arguments)
        params_sampled.update(lens_source_sampled)

        if 'zlens' in params_sampled.keys():
            pyhalo_dynamic = pyHaloDynamic(zlens, zsource)
        else:
            if pyhalo_dynamic is None:
                pyhalo_dynamic = pyHaloDynamic(zlens, zsource)

        ######## Sample keyword arguments for the macromodel ##########
        macromodel, macro_samples, constrain_params, opt_routine = \
            load_powerlaw_ellipsoid_macromodel(zlens, prior_list_macromodel, kwargs_macro_ref)
        params_sampled.update(macro_samples)

        ######## Sample keyword arguments for the background source ##########
        background_quasar, source_samples = load_background_quasar(prior_list_source,
                                                                   keyword_arguments)
        params_sampled.update(source_samples)

        ################## Set up the data to fit ####################
        data_to_fit = load_data_to_fit(keyword_arguments)
        # import matplotlib.pyplot as plt
        # plt.scatter(data_to_fit.x, data_to_fit.y)
        # plt.show()

        ################ Get the optimization settings ################
        optimization_settings = load_optimization_settings(keyword_arguments)

        ################ Perform the fit ################
        lens_system = QuadLensSystem(macromodel, zsource, background_quasar,
                                     None, None)
        if initialize:
            lens_system.initialize(data_to_fit_init, opt_routine, constrain_params)
            kwargs_macro_ref = lens_system.macromodel.components[0].kwargs

        kwargs_rendering['cone_opening_angle'] = kwargs_rendering['opening_angle_factor'] * \
                                                 theta_E_approx
        optimization_settings['initial_pso'] = initialize

        dynamic_opt = DynamicOptimization(lens_system, pyhalo_dynamic, kwargs_rendering,
                                          **optimization_settings)
        kwargs_lens_fit, lensModel_fit, _ = \
            dynamic_opt.optimize(data_to_fit, opt_routine=opt_routine,
                 constrain_params=constrain_params, verbose=keyword_arguments['verbose'])

        flux_ratios_fit, _ = lens_system.quasar_magnification(data_to_fit.x, data_to_fit.y,
                                            lensModel_fit, kwargs_lens_fit)
        flux_ratios_fit = np.round(flux_ratios_fit, 5)

        comp1 = kwargs_e1e2_to_polar(lens_system.macromodel.components[0].kwargs[0])
        comp2 = kwargs_gamma1gamma2_to_polar(lens_system.macromodel.components[0].kwargs[1])
        kwargs_macro_new = {}
        for key in comp1.keys():
            kwargs_macro_new[key] = comp1[key]
        for key in comp2.keys():
            kwargs_macro_new[key] = comp2[key]
        kwargs_macro.append(kwargs_macro_new)

        if keyword_arguments['verbose']:
            print('flux_ratios_fit:', flux_ratios_fit)
            print('n remaining: ', keyword_arguments['Nsamples'] - (i + 1))

        header = ''
        for name in params_sampled.keys():
            header += name + ' '
            parameters.append(params_sampled[name])
        parameters = np.array(parameters)

        if fluxes_computed is None and parameters_sampled is None:
            fluxes_computed = flux_ratios_fit
            parameters_sampled = parameters
        else:

            fluxes_computed = np.vstack((fluxes_computed, flux_ratios_fit))
            parameters_sampled = np.vstack((parameters_sampled, parameters))

        if (i+1) % readout_steps == 0:

            readout(readout_path, kwargs_macro, fluxes_computed, parameters_sampled, header, write_header)

    return

chain_ID = 'test_submit'
output_path = os.getenv('HOME') + '/data/sims/'+chain_ID + '/'
paramdictionary_folder_path = os.getenv('HOME') + '/data/'
run(2, output_path, paramdictionary_folder_path + chain_ID)















