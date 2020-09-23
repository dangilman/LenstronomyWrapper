import os

from lenstronomywrapper.Sampler.utilities import *
from lenstronomywrapper.Utilities.misc import create_directory
from lenstronomywrapper.LensSystem.quad_lens import QuadLensSystem
from lenstronomywrapper.Utilities.data_util import approx_theta_E

from lenstronomywrapper.Utilities.parameter_util import kwargs_e1e2_to_polar, kwargs_gamma1gamma2_to_polar

from pyHalo.pyhalo_dynamic import pyHaloDynamic
from pyHalo.pyhalo import pyHalo

from time import time

from lenstronomywrapper.Optimization.quad_optimization.dynamic import DynamicOptimization
from lenstronomywrapper.Optimization.quad_optimization.hierarchical import HierarchicalOptimization
from lenstronomywrapper.Utilities.misc import write_lensdata

def run(job_index, chain_ID, output_path, path_to_folder,
        test_mode=False):

    t_start = time()

    output_path += chain_ID + '/'
    path_to_folder += chain_ID

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

    write_header = True
    write_mode = 'w'
    if os.path.exists(fname_fluxes):
        fluxes_computed = np.loadtxt(fname_fluxes)
        N_computed = int(fluxes_computed.shape[0])
        write_header = False
        write_mode = 'a'
    n_run = Nsamples - N_computed

    if n_run <= 0:
        print('job index '+str(job_index) + ' finished.')
        return
    else:
        print('running job index '+str(job_index) + ', '+str(n_run) + ' realizations remaining')

    ############ Extract and organize key word arguments ###############
    prior_list_realization, \
    prior_list_macromodel, \
    prior_list_source, \
    prior_list_cosmo = \
        build_priors(keyword_arguments['params_to_vary'])

    data_to_fit_init = load_data_to_fit(keyword_arguments)

    write_lensdata(readout_path + 'lensdata.txt', data_to_fit_init.x,
                   data_to_fit_init.y, data_to_fit_init.m,
                   [0.] * 4)

    theta_E_approx = approx_theta_E(data_to_fit_init.x, data_to_fit_init.y)

    ############################ EVERYTHING BELOW THIS IS SAMPLED IN A FOR LOOP ############################

    pyhalo = None
    kwargs_macro = []
    initialize = True
    kwargs_macro_ref = None
    fluxes_computed = None
    parameters_sampled = None

    adaptive_mag = keyword_arguments['adaptive_mag']

    counter = 0
    while counter < n_run:

        params_sampled = {}
        parameters = []

        ######## Sample keyword arguments for the substructure realization ##########

        kwargs_rendering, realization_samples = realization_keywords(keyword_arguments, prior_list_realization)
        params_sampled.update(realization_samples)

        ######## Sample keyword arguments for the lensing volume ##########
        zlens, zsource, lens_source_sampled = load_lens_source(prior_list_cosmo, keyword_arguments)
        params_sampled.update(lens_source_sampled)

        ######## Sample keyword arguments for the macromodel ##########

        macromodel, macro_samples, constrain_params, opt_routine = \
            load_powerlaw_ellipsoid_macromodel(zlens, prior_list_macromodel, kwargs_macro_ref,
                                               keyword_arguments['secondary_lens_components'])
        params_sampled.update(macro_samples)

        ######## Sample keyword arguments for the background source ##########
        background_quasar, source_samples = load_background_quasar(prior_list_source,
                                                                   keyword_arguments)
        params_sampled.update(source_samples)

        ################## Set up the data to fit ####################
        data_to_fit = load_data_to_fit(keyword_arguments)

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

        assert 'routine' in keyword_arguments['keywords_optimizer'].keys()
        if test_mode:
            keyword_arguments['verbose'] = True

        if keyword_arguments['verbose']:
            for key in params_sampled.keys():
                print(key + ': ' + str(params_sampled[key]))
            print('zlens', zlens)
            print('zsource', zsource)
            print('Einstein radius', theta_E_approx)

        if keyword_arguments['keywords_optimizer']['routine'] == 'dynamic':

            if 'zlens' in params_sampled.keys():
                pyhalo = pyHaloDynamic(zlens, zsource)
            else:
                if pyhalo is None:
                    pyhalo = pyHaloDynamic(zlens, zsource)

            dynamic_opt = DynamicOptimization(lens_system, pyhalo,
                                              kwargs_rendering, **optimization_settings)
            kwargs_lens_fit, lensModel_fit, _ = \
            dynamic_opt.optimize(
                data_to_fit, opt_routine=opt_routine,
                constrain_params=constrain_params, verbose=keyword_arguments['verbose']
            )

        elif keyword_arguments['keywords_optimizer']['routine'] == 'hierarchical':

            if 'zlens' in params_sampled.keys():
                kwargs_hmf = keyword_arguments['realization_kwargs']['kwargs_halo_mass_function']
                pyhalo = pyHalo(zlens, zsource, kwargs_halo_mass_function=kwargs_hmf)
            else:
                kwargs_hmf = keyword_arguments['realization_kwargs']['kwargs_halo_mass_function']
                if pyhalo is None:
                    pyhalo = pyHalo(zlens, zsource, kwargs_halo_mass_function=kwargs_hmf)

            realization_initial = pyhalo.render(keyword_arguments['realization_type'],
                                        kwargs_rendering)[0]
            lens_system = QuadLensSystem.addRealization(lens_system, realization_initial)

            if 'settings_class' in keyword_arguments['keywords_optimizer'].keys():
                settings_class = keyword_arguments['keywords_optimizer']['settings_class']
            else:
                print('WARNING, USUING A DEFAULT SETTING FOR THE LENS MODELING COMPUTATIONS. '
                      'CHECK WITH DANIEL IF THIS IS OK BEFORE CONTINUING!!!')
                settings_class = 'default'

            if keyword_arguments['verbose']:
                print('realization has '+str(len(realization_initial.halos))+' halos in total')

            hierarchical_opt = HierarchicalOptimization(lens_system, settings_class=settings_class)
            kwargs_lens_fit, lensModel_fit, _ = hierarchical_opt.optimize(
                data_to_fit, opt_routine, constrain_params, keyword_arguments['verbose']
            )

        else:
            raise Exception('optimization routine '+ keyword_arguments['keywords_optimizer']['routine']
                            + 'not recognized.')

        flux_ratios_fit, blended = lens_system.quasar_magnification(
            data_to_fit.x, data_to_fit.y, lensModel_fit,
            kwargs_lens_fit, enforce_unblended=True,
            adaptive=adaptive_mag, verbose=keyword_arguments['verbose']
        )

        if test_mode:
            import matplotlib.pyplot as plt

            ax = plt.gca()
            ax.scatter(data_to_fit.x, data_to_fit.y)
            ax.set_xlim(-1.5*theta_E_approx, 1.5*theta_E_approx)
            ax.set_ylim(-1.5 * theta_E_approx, 1.5 * theta_E_approx)
            ax.set_aspect('equal')
            plt.show()

            lens_system.plot_images(data_to_fit.x, data_to_fit.y, adaptive=adaptive_mag)
            plt.show()
            a = input('continue')

        if blended:
            print('images are blended together')
            continue

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
            print('n remaining: ', keyword_arguments['Nsamples'] - (counter + 1))

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

        if (counter+1) % readout_steps == 0:
            t_end = time()
            t_ellapsed = t_end - t_start
            sampling_rate = fluxes_computed.shape[0] / t_ellapsed
            readout(readout_path, kwargs_macro, fluxes_computed, parameters_sampled,
                    header, write_header, write_mode, sampling_rate)
            fluxes_computed, parameters_sampled = None, None
            kwargs_macro = []
            write_mode = 'a'
            write_header = False
            t_start = time()

        counter += 1

    return
