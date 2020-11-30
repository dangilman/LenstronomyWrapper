import os

from lenstronomywrapper.Sampler.utilities import *
from lenstronomywrapper.Utilities.misc import create_directory
from lenstronomywrapper.LensSystem.quad_lens import QuadLensSystem
from lenstronomywrapper.Utilities.data_util import approx_theta_E

from lenstronomywrapper.Utilities.parameter_util import kwargs_e1e2_to_polar, kwargs_gamma1gamma2_to_polar
from lenstronomywrapper.LensSystem.BackgroundSource.double_gaussian import DoubleGaussian
from lenstronomywrapper.LensSystem.local_image_quad import LocalImageQuad
from pyHalo.pyhalo import pyHalo

from time import time

from scipy.stats.kde import gaussian_kde

from lenstronomywrapper.Optimization.quad_optimization.dynamic import DynamicOptimization
from lenstronomywrapper.Optimization.quad_optimization.hierarchical import HierarchicalOptimization
from lenstronomywrapper.Optimization.quad_optimization.hierarchical_local import HierarchicalOptimizationLocal
from lenstronomywrapper.Utilities.misc import write_lensdata


def run_1131(job_index, chain_ID, output_path, path_to_folder,
                    test_mode=False):
    t_start = time()

    output_path += chain_ID + '/'
    path_to_folder += chain_ID

    if not os.path.exists(output_path):
        print('creating directory ' + output_path)
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

    fname_fluxes = readout_path + 'fluxes_NL.txt'

    write_header = True
    write_mode = 'w'

    if os.path.exists(fname_fluxes):
        fluxes_computed = np.loadtxt(fname_fluxes)
        N_computed = int(fluxes_computed.shape[0])
        write_header = False
        write_mode = 'a'
    n_run = Nsamples - N_computed

    if n_run <= 0:
        print('job index ' + str(job_index) + ' finished.')
        return
    else:
        print('running job index ' + str(job_index) + ', ' + str(n_run) + ' realizations remaining')

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

        macromodel, macro_samples, constrain_params = \
            load_powerlaw_ellipsoid_macromodel(zlens, prior_list_macromodel, kwargs_macro_ref,
                                               keyword_arguments['secondary_lens_components'])
        params_sampled.update(macro_samples)

        ######## Sample keyword arguments for the background source ##########
        background_quasar, source_samples = load_background_quasar(prior_list_source,
                                                                     keyword_arguments)
        params_sampled.update(source_samples)
        kwargs_quasar = {'center_x': 0., 'center_y': 0., 'source_fwhm_pc': 1.}
        background_quasar = Quasar(kwargs_quasar)

        ################## Set up the data to fit ####################
        data_to_fit = load_data_to_fit(keyword_arguments)

        ################ Get the optimization settings ################
        optimization_settings = load_optimization_settings(keyword_arguments)

        ################ Perform a fit with only a smooth model ################
        optimization_routine = keyword_arguments['optimization_routine']
        lens_system = QuadLensSystem(macromodel, zsource, background_quasar,
                                     None, None)
        lens_system.initialize(data_to_fit_init, opt_routine=optimization_routine,
                constrain_params=constrain_params)
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

        if 'zlens' in params_sampled.keys():
            kwargs_hmf = keyword_arguments['realization_kwargs']['kwargs_halo_mass_function']
            pyhalo = pyHalo(zlens, zsource, kwargs_halo_mass_function=kwargs_hmf)
        else:
            kwargs_hmf = keyword_arguments['realization_kwargs']['kwargs_halo_mass_function']
            if pyhalo is None:
                pyhalo = pyHalo(zlens, zsource, kwargs_halo_mass_function=kwargs_hmf)

        deltas, delta_hessian = None, None
        readout_macro = True

        if keyword_arguments['keywords_optimizer']['routine'] == 'hierarchical':

            if 'log_mlow' in realization_samples.keys():
                kwargs_rendering['log_mass_sheet_min'] = max(kwargs_rendering['log_mass_sheet_min'],
                                                             realization_samples['log_mlow'])

            realization_initial = pyhalo.render(keyword_arguments['realization_type'],
                                                kwargs_rendering)[0]
            lens_system = QuadLensSystem.addRealization(lens_system, realization_initial)

            if 'settings_class' in keyword_arguments['keywords_optimizer'].keys():
                settings_class = keyword_arguments['keywords_optimizer']['settings_class']
                if settings_class == 'custom':
                    assert 'kwargs_settings_class' in keyword_arguments['keywords_optimizer'].keys()
                    kwargs_settings_class = keyword_arguments['keywords_optimizer']['kwargs_settings_class']
                else:
                    kwargs_settings_class = None
            else:
                print('WARNING, USUING A DEFAULT SETTING FOR THE LENS MODELING COMPUTATIONS. '
                      'CHECK WITH DANIEL IF THIS IS OK BEFORE CONTINUING!!!')
                settings_class = 'default'
                kwargs_settings_class = None

            if keyword_arguments['verbose']:
                print('realization has ' + str(len(realization_initial.halos)) + ' halos in total')

            hierarchical_opt = HierarchicalOptimization(lens_system, settings_class=settings_class,
                                                        kwargs_settings_class=kwargs_settings_class)
            kwargs_lens_fit, lensModel_fit, _ = hierarchical_opt.optimize(
                data_to_fit, optimization_routine, constrain_params, keyword_arguments['verbose']
            )

            srcx_fit, srcy_fit = lens_system.source_centroid_x, lens_system.source_centroid_y

            kwargs_src_single = {'center_x': srcx_fit, 'center_y': srcy_fit,
                                 'source_fwhm_pc': source_samples['source_fwhm_pc']}


            kwargs_src_double = {'center_x': srcx_fit, 'center_y': srcy_fit,
                                 'source_fwhm_pc': source_samples['source_fwhm_pc'],
                                 'dx': source_samples['dx_source_2'],
                                 'dy': source_samples['dy_source_2'],
                                 'size_scale': source_samples['size_scale_2'],
                                 'amp_scale': source_samples['amp_scale_2']}

            aperture_diameter = 0.77
            # 2.5 because adaptive mag rescales the size by 2.5
            aperture_radius = aperture_diameter / 2 / 2.5

            quasar_single = Quasar(kwargs_src_single, grid_rmax=aperture_radius)
            quasar_single.setup(lens_system.pc_per_arcsec_zsource)
            quasar_dbl = DoubleGaussian(kwargs_src_double, grid_rmax=aperture_radius)
            quasar_dbl.setup(lens_system.pc_per_arcsec_zsource)

            magnification_function_single = quasar_single.magnification
            magnification_function_dbl = quasar_dbl.magnification
            magnification_function_kwargs = {'xpos': data_to_fit.x, 'ypos': data_to_fit.y,
                                             'lensModel': lensModel_fit, 'kwargs_lens': kwargs_lens_fit, 'normed': True,
                                             'enforce_unblended': True, 'adaptive': adaptive_mag,
                                             'verbose': keyword_arguments['verbose']}

        else:
            raise Exception('optimization routine ' + keyword_arguments['keywords_optimizer']['routine']
                            + 'not recognized.')

        flux_ratios_fit_single, blended_single = magnification_function_single(**magnification_function_kwargs)
        flux_ratios_fit_double, blended_dbl = magnification_function_dbl(**magnification_function_kwargs)

        if blended_single:
            blended = True
        else:
            blended = False

        if test_mode:
            import matplotlib.pyplot as plt
            if deltas is not None:
                for delta in deltas:
                    print(delta)

            quasar_single.plot_images(data_to_fit.x, data_to_fit.y,
                                  lensModel_fit, kwargs_lens_fit,
                                  normed=True, adaptive=True)
            plt.show()
            a=input('continue')
            quasar_dbl.plot_images(data_to_fit.x, data_to_fit.y,
                                     lensModel_fit, kwargs_lens_fit,
                                     normed=True, adaptive=True)
            plt.show()
            a = input('continue')

        if blended:
            print('images are blended together')
            continue

        flux_ratios_fit_single = np.round(flux_ratios_fit_single, 5)
        flux_ratios_fit_double = np.round(flux_ratios_fit_double, 5)

        if readout_macro:
            comp1 = kwargs_e1e2_to_polar(lens_system.macromodel.components[0].kwargs[0])
            comp2 = kwargs_gamma1gamma2_to_polar(lens_system.macromodel.components[0].kwargs[1])
            kwargs_macro_new = {}
            for key in comp1.keys():
                kwargs_macro_new[key] = comp1[key]
            for key in comp2.keys():
                kwargs_macro_new[key] = comp2[key]
            kwargs_macro.append(kwargs_macro_new)

        if keyword_arguments['verbose']:
            print('flux_ratios_fit single:', flux_ratios_fit_single)
            print('flux_ratios_fit mid double:', flux_ratios_fit_double)
            print('n remaining: ', keyword_arguments['Nsamples'] - (counter + 1))

        header = ''
        for name in params_sampled.keys():
            header += name + ' '
            parameters.append(params_sampled[name])
        parameters = np.array(parameters)

        if fluxes_computed is None and parameters_sampled is None:

            fluxes_computed_single = flux_ratios_fit_single
            fluxes_computed_double = flux_ratios_fit_double
            parameters_sampled = parameters

        else:

            fluxes_computed_single = np.vstack((fluxes_computed_single, flux_ratios_fit_single))
            fluxes_computed_double = np.vstack((fluxes_computed_double, flux_ratios_fit_double))
            parameters_sampled = np.vstack((parameters_sampled, parameters))

        if (counter + 1) % readout_steps == 0:
            t_end = time()
            t_ellapsed = t_end - t_start
            sampling_rate = fluxes_computed_single.shape[0] / t_ellapsed

            readout(readout_path, kwargs_macro, fluxes_computed_single, parameters_sampled,
                    header, write_header, write_mode, sampling_rate, readout_macro,
                    delta_hessian, flux_file_extension='_single_source')
            readout(readout_path, None, fluxes_computed_double, None,
                    None, None, write_mode, None, None, None,
                    readout_flux_only=True, flux_file_extension='_double_source')

            fluxes_computed_single, fluxes_computed_double, parameters_sampled = None, None, None
            kwargs_macro = []
            write_mode = 'a'
            write_header = False
            t_start = time()

        counter += 1

    return
