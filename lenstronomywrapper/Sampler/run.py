import os

from lenstronomywrapper.Sampler.utilities import *
from lenstronomywrapper.Utilities.misc import create_directory
from lenstronomywrapper.LensSystem.quad_lens import QuadLensSystem
from lenstronomywrapper.Utilities.data_util import approx_theta_E
import dill
from pyHalo.single_realization import add_core_collapsed_subhalos
from lenstronomywrapper.Utilities.parameter_util import kwargs_e1e2_to_polar, kwargs_gamma1gamma2_to_polar

from lenstronomywrapper.LensSystem.local_image_quad import LocalImageQuad
from pyHalo.pyhalo import pyHalo

from time import time

from scipy.stats.kde import gaussian_kde

from lenstronomywrapper.Optimization.quad_optimization.dynamic import DynamicOptimization
from lenstronomywrapper.Optimization.quad_optimization.hierarchical import HierarchicalOptimization
from lenstronomywrapper.Optimization.quad_optimization.hierarchical_local import HierarchicalOptimizationLocal
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
    delta_kappa = None
    delta_gamma1 = None
    delta_gamma2 = None
    hessian_kde = None

    adaptive_mag = keyword_arguments['adaptive_mag']

    if 'save_best_realization' in keyword_arguments.keys():
        assert 'fluxes' in keyword_arguments.keys(), "must specify target fluxes if saving best realizations"
        save_best_realization = keyword_arguments['save_best_realization']
        if os.path.exists(readout_path + 'best_realization'):
            f = open(readout_path + 'best_realization', 'rb')
            best_realization = dill.load(f)
            f.close()
            current_best_statistic = best_realization.statistic
            print('starting from best statistic of: ', current_best_statistic)

        else:
            current_best_statistic = 1e+6

    else:
        save_best_realization = False
    readout_best = False

    if 'enforce_unblended' in keyword_arguments.keys():
        enforce_unblended = keyword_arguments['enforce_unblended']
    else:
        enforce_unblended = True

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
                                               keyword_arguments['secondary_lens_components'], keyword_arguments)
        params_sampled.update(macro_samples)

        ######## Sample keyword arguments for the background source ##########
        background_quasar, source_samples = load_background_quasar(prior_list_source,
                                                                   keyword_arguments)
        params_sampled.update(source_samples)

        ################## Set up the data to fit ####################
        data_to_fit = load_data_to_fit(keyword_arguments)

        ################ Get the optimization settings ################
        optimization_settings = load_optimization_settings(keyword_arguments)

        ################ Perform a fit with only a smooth model ################
        optimization_routine = keyword_arguments['optimization_routine']
        lens_system = QuadLensSystem(macromodel, zsource, background_quasar,
                                     None, None)
        lens_system.initialize(data_to_fit_init, optimization_routine, constrain_params)
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
        #print(
        if 'zlens' in params_sampled.keys():
            kwargs_hmf = keyword_arguments['realization_kwargs']['kwargs_halo_mass_function']
            pyhalo = pyHalo(zlens, zsource, kwargs_halo_mass_function=kwargs_hmf)
        else:
            kwargs_hmf = keyword_arguments['realization_kwargs']['kwargs_halo_mass_function']
            if pyhalo is None:
                pyhalo = pyHalo(zlens, zsource, kwargs_halo_mass_function=kwargs_hmf)

        deltas, delta_hessian = None, None
        readout_macro = True
        if keyword_arguments['keywords_optimizer']['routine'] == 'local':

            readout_macro = False
            source_x, source_y = source_samples['source_x'], source_samples['source_y']
            if 'local_image_keywords' not in keyword_arguments.keys():
                raise Exception('when using optimizer class local, must specify lens_model_list, kwargs_macro, '
                                'redshift_list, and macro_indicies_fixed in local_image_keywords')

            macro_indicies_fixed, hessian_samples, lensmodel_macro, kwargs_lens_macro, \
            angular_scale, samples_local_image, kwargs_opt = load_local_image_keywords(keyword_arguments['local_image_keywords'], lens_system)

            if hessian_kde is None:
                hessian_kde = [gaussian_kde(dataset=hessian_samples[0], bw_method='silverman'),
                               gaussian_kde(dataset=hessian_samples[1], bw_method='silverman'),
                               gaussian_kde(dataset=hessian_samples[2], bw_method='silverman'),
                               gaussian_kde(dataset=hessian_samples[3], bw_method='silverman')]

            hessian_constraints = []
            for h in hessian_kde:
                (fxx, fxy, fyy) = h.resample(1).T[0]
                hess = np.array([fxx, fxy, fxy, fyy])
                hessian_constraints.append(hess)

            params_sampled.update(samples_local_image)

            lens_system = LocalImageQuad(data_to_fit.x, data_to_fit.y, source_x, source_y,
                                              lensmodel_macro, kwargs_lens_macro, zlens, zsource,
                                              macro_indicies_fixed, pyhalo.cosmology)
            opt = HierarchicalOptimizationLocal(lens_system)
            deltas = opt.fit(angular_scale, hessian_constraints, verbose=False, **kwargs_opt)

            background_quasar.setup(center_x=source_x, center_y=source_y)
            magnification_function_kwargs = {'source_model': background_quasar,
                                            'adaptive': adaptive_mag, 'verbose': keyword_arguments['verbose']}
            magnification_function = lens_system.magnification

        elif keyword_arguments['keywords_optimizer']['routine'] == 'hierarchical':

            realization_initial = pyhalo.render(keyword_arguments['realization_type'],
                                        kwargs_rendering)[0]

            if 'f_core_collapsed' in realization_samples.keys():
                f = realization_samples['f_core_collapsed']
                realization_initial = add_core_collapsed_subhalos(f, realization_initial)

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
                print('realization has '+str(len(realization_initial.halos))+' halos in total')

            if 'check_bad_fit' in keyword_arguments.keys():
                check_bad_fit = keyword_arguments['check_bad_fit']
            else:
                check_bad_fit = False

            hierarchical_opt = HierarchicalOptimization(lens_system, settings_class=settings_class,
                                                        kwargs_settings_class=kwargs_settings_class)
            kwargs_lens_fit, lensModel_fit, _ = hierarchical_opt.optimize(
                data_to_fit, optimization_routine, constrain_params, keyword_arguments['verbose'],
                check_bad_fit=check_bad_fit
            )
            if kwargs_lens_fit is None:
                continue

            if 'relative_angles' in keyword_arguments.keys():
                assert len(keyword_arguments['relative_angles']) == 4
                relative_angles = keyword_arguments['relative_angles']
                assert 'grid_axis_ratio' in keyword_arguments.keys()
                grid_axis_ratio = keyword_arguments['grid_axis_ratio']
            else:
                relative_angles = None
                grid_axis_ratio = 1.

            magnification_function = lens_system.quasar_magnification
            magnification_function_kwargs = {'x': data_to_fit.x, 'y': data_to_fit.y,
                         'lens_model': lensModel_fit, 'kwargs_lensmodel': kwargs_lens_fit, 'normed': True,
                             'enforce_unblended': enforce_unblended, 'adaptive': adaptive_mag,
                                             'verbose': keyword_arguments['verbose'],
                                             'relative_angles': relative_angles,
                                             'grid_axis_ratio': grid_axis_ratio}

        else:
            raise Exception('optimization routine '+ keyword_arguments['keywords_optimizer']['routine']
                            + 'not recognized.')

        flux_ratios_fit, blended = magnification_function(**magnification_function_kwargs)

        if test_mode:
            import matplotlib.pyplot as plt
            if deltas is not None:
                for delta in deltas:
                    print(delta)

            ax = plt.gca()
            ax.scatter(data_to_fit.x, data_to_fit.y)
            ax.set_xlim(-1.9*theta_E_approx, 1.9*theta_E_approx)
            ax.set_ylim(-1.9 * theta_E_approx, 1.9 * theta_E_approx)
            ax.set_aspect('equal')
            plt.show()

            lens_system.plot_images(data_to_fit.x, data_to_fit.y, adaptive=adaptive_mag)
            plt.show()

            _x = _y = np.linspace(-1.5, 1.5, 100)
            xx, yy = np.meshgrid(_x, _y)
            shape0 = xx.shape
            mag_surface = lensModel_fit.magnification(xx.ravel(), yy.ravel(),
                                        kwargs_lens_fit).reshape(shape0)

            plt.imshow(np.log10(mag_surface), extent=[-1.5, 1.5, -1.5, 1.5], origin='lower')
            plt.scatter(data_to_fit.x, data_to_fit.y, color='k')
            plt.show()

            a = input('continue')

        if blended and enforce_unblended:
            print('images are blended together')
            continue

        flux_ratios_fit = np.round(flux_ratios_fit, 5)

        if save_best_realization:
            fluxes_measured = np.array(keyword_arguments['fluxes'])
            df = flux_ratios_fit[1:]/flux_ratios_fit[0] - fluxes_measured[1:]/fluxes_measured[0]
            new_statistic = np.sum(np.sqrt(df ** 2))
            if test_mode or keyword_arguments['verbose']:
                print('new statistic: ', new_statistic)
                print('current best statistic: ', current_best_statistic)
            if new_statistic < current_best_statistic:
                readout_best = True
                print('storing new realization...')
                current_best_statistic = new_statistic
                best_realization = SavedRealization(lensModel_fit, kwargs_lens_fit, flux_ratios_fit,
                                                    new_statistic, parameters)

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
            if deltas is not None:
                delta_kappa = deltas[0]
                delta_gamma1 = deltas[1]
                delta_gamma2 = deltas[2]

        else:

            fluxes_computed = np.vstack((fluxes_computed, flux_ratios_fit))
            parameters_sampled = np.vstack((parameters_sampled, parameters))
            if deltas is not None:
                delta_kappa = np.vstack((delta_kappa, deltas[0]))
                delta_gamma1 = np.vstack((delta_gamma1, deltas[1]))
                delta_gamma2 = np.vstack((delta_gamma2, deltas[2]))

        if (counter+1) % readout_steps == 0:
            t_end = time()
            t_ellapsed = t_end - t_start
            sampling_rate = fluxes_computed.shape[0] / t_ellapsed

            if delta_kappa is not None:
                delta_hessian = (delta_kappa, delta_gamma1, delta_gamma2)
            else:
                delta_hessian = None

            readout(readout_path, kwargs_macro, fluxes_computed, parameters_sampled,
                    header, write_header, write_mode, sampling_rate, readout_macro, delta_hessian)
            fluxes_computed, parameters_sampled = None, None
            delta_kappa, delta_gamma1, delta_gamma2 = None, None, None
            kwargs_macro = []
            write_mode = 'a'
            write_header = False
            t_start = time()

            if save_best_realization and readout_best:
                f = open(readout_path + 'best_realization', 'wb')
                dill.dump(best_realization, f)
                f.close()

        counter += 1

    return

class SavedRealization(object):

    def __init__(self, lensmodel_instance, kwargs_lens_fit, fluxes_modeled,
                 statistic, params):

        self.lensmodel = lensmodel_instance
        self.kwargs = kwargs_lens_fit
        self.fluxes_modeled = fluxes_modeled
        self.statistic = statistic
        self.params = params
