import os

from lenstronomywrapper.Sampler.utilities import *
from lenstronomywrapper.Utilities.misc import create_directory
from lenstronomywrapper.LensSystem.quad_lens import QuadLensSystem
from lenstronomywrapper.Utilities.data_util import approx_theta_E
import dill
from lenstronomywrapper.Utilities.parameter_util import kwargs_e1e2_to_polar, kwargs_gamma1gamma2_to_polar
from time import time
from lenstronomywrapper.Optimization.quad_optimization.hierarchical import HierarchicalOptimization
from lenstronomywrapper.Utilities.misc import write_lensdata
from pyHalo.preset_models import WDMLovell2020, CDM
from sidmpy.Profiles.coreTNFW_deflection_angle import CoreTNFWDeflection
from sidmpy.sidmpy import solve_rho_with_interpolation
from pyHalo.realization_extensions import RealizationExtensions
from sidmpy.CrossSections.tchannel import TChannel
from sidmpy.core_collapse_timescale import evolution_timescale_scattering_rate
from sidmpy.sidmpy import solve_sigmav_with_interpolation

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

    keywords_master = load_keywords(path_to_folder, job_index)
    if test_mode:
        keywords_master['verbose'] = True

    readout_path = output_path + 'chain_' + str(job_index) + '/'
    if not os.path.exists(readout_path):
        create_directory(readout_path)

    readout_steps = keywords_master['readout_steps']
    N_computed = 0

    write_header = True
    write_mode = 'w'
    if os.path.exists(readout_path + 'fluxes.txt'):
        fluxes_computed = np.loadtxt(readout_path + 'fluxes.txt')
        N_computed = int(fluxes_computed.shape[0])
        write_header = False
        write_mode = 'a'

    Nsamples = keywords_master['Nsamples']
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
        build_priors(keywords_master['params_to_vary'])

    data_to_fit_init = LensedQuasar(keywords_master['x_image'], keywords_master['y_image'],
                                    keywords_master['fluxes'])
    write_lensdata(readout_path + 'lensdata.txt', data_to_fit_init.x,
                   data_to_fit_init.y, data_to_fit_init.m,
                   [0.] * 4)

    theta_E_approx = approx_theta_E(data_to_fit_init.x, data_to_fit_init.y)

    ############################ EVERYTHING BELOW THIS IS SAMPLED IN A FOR LOOP ############################

    kwargs_macro = []
    initial_pso = True
    kwargs_macro_ref = None
    fluxes_computed = None
    fluxes_computed_2 = None
    parameters_sampled = None

    if 'save_best_realization' in keywords_master.keys():
        assert 'fluxes' in keywords_master.keys(), "must specify target fluxes if saving best realizations"
        save_best_realization = keywords_master['save_best_realization']
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

    counter = 0
    while counter < n_run:

        setup = simulation_setup(keywords_master, prior_list_realization, prior_list_cosmo,
                                 prior_list_macromodel, prior_list_source, kwargs_macro_ref)

        (kwargs_rendering,
         realization_samples,
         zlens,
         zsource,
         lens_source_sampled, \
        macromodel,
         macro_samples,
         constrain_params,
         source_samples,
         data_to_fit, \
        optimization_settings,
         optimization_routine,
         params_sampled) = setup

        kwargs_rendering['cone_opening_angle'] = kwargs_rendering['opening_angle_factor'] * \
                                                 theta_E_approx
        cross_section_norm = kwargs_rendering['norm']
        cross_section_v = kwargs_rendering['v0']
        cross_section_type = 'TCHANNEL'
        deflection_angle_computation = CoreTNFWDeflection()
        kwargs_cross_section = {'norm': cross_section_norm, 'v_ref': cross_section_v}
        kwargs_sidm = {'cross_section_type': cross_section_type, 'kwargs_cross_section': kwargs_cross_section,
                       'SIDM_rhocentral_function': solve_rho_with_interpolation,
                       'numerical_deflection_angle_class': deflection_angle_computation}
        kwargs_rendering.update(kwargs_sidm)

        optimization_settings['initial_pso'] = initial_pso

        assert 'routine' in keywords_master['keywords_optimizer'].keys()

        if keywords_master['verbose']:
            for key in params_sampled.keys():
                print(key + ': ' + str(params_sampled[key]))
            print('zlens', zlens)
            print('zsource', zsource)
            print('Einstein radius', theta_E_approx)

        readout_macro = True

        if 'preset_model' in keywords_master.keys():
            if keywords_master['preset_model'] == 'WDMLovell2020':
                realization_initial = WDMLovell2020(zlens, zsource, **kwargs_rendering)
            elif keywords_master['preset_model'] == 'CDM':
                realization_initial = CDM(zlens, zsource, **kwargs_rendering)
            else:
                raise Exception('no other preset model recognized')
        else:
            raise Exception('must specify preset model')

        ext = RealizationExtensions(realization_initial)
        cross_section = TChannel(**kwargs_cross_section)
        inds = ext.find_core_collapsed_halos(evolution_timescale_scattering_rate, solve_sigmav_with_interpolation,
                                             cross_section, t_sub=10, t_field=100)
        print('n halos core collapsed: ', len(inds)/len(realization_initial.halos))
        realization_initial = ext.add_core_collapsed_halos(inds, log_slope_halo=3., x_core_halo=0.05)

        lens_system = QuadLensSystem.shift_background_auto(data_to_fit, macromodel, zsource,
                               realization_initial, None, particle_swarm_init=True,
                                opt_routine=optimization_routine, constrain_params=constrain_params,
                                                           verbose=keywords_master['verbose'])

        settings_class = keywords_master['keywords_optimizer']['settings_class']

        if keywords_master['verbose']:
            print('realization has '+str(len(realization_initial.halos))+' halos in total')
        if 'check_bad_fit' in keywords_master.keys():
            check_bad_fit = keywords_master['check_bad_fit']
        else:
            check_bad_fit = False

        hierarchical_opt = HierarchicalOptimization(lens_system, settings_class=settings_class)
        kwargs_lens_fit, lensModel_fit, _ = hierarchical_opt.optimize(
            data_to_fit, optimization_routine, constrain_params, keywords_master['verbose'],
            check_bad_fit=check_bad_fit)

        if kwargs_lens_fit is None:
            continue

        if 'grid_axis_ratio' in keywords_master.keys():
            grid_axis_ratio = keywords_master['grid_axis_ratio']
        else:
            grid_axis_ratio = 0.5

        if 'grid_rmax' in keywords_master:
            grid_rmax = keywords_master['grid_rmax']
        else:
            grid_rmax = None

        magnification_function = lens_system.quasar_magnification
        magnification_function_kwargs = {'x': data_to_fit.x, 'y': data_to_fit.y,
                      'source_fwhm_pc': source_samples['source_fwhm_pc'],
                     'lens_model': lensModel_fit, 'kwargs_lensmodel': kwargs_lens_fit, 'normed': True,
                         'grid_axis_ratio': grid_axis_ratio, 'grid_rmax': grid_rmax,
                     'grid_resolution_rescale': 2.5, 'source_light_model': 'SINGLE_GAUSSIAN'}
        flux_ratios_fit = magnification_function(**magnification_function_kwargs)

        magnification_function = lens_system.quasar_magnification
        magnification_function_kwargs = {'x': data_to_fit.x, 'y': data_to_fit.y,
                                         'source_fwhm_pc': source_samples['source_fwhm_pc_2'],
                                         'lens_model': lensModel_fit, 'kwargs_lensmodel': kwargs_lens_fit,
                                         'normed': True,
                                         'grid_axis_ratio': grid_axis_ratio, 'grid_rmax': grid_rmax,
                                         'grid_resolution_rescale': 2.5, 'source_light_model': 'SINGLE_GAUSSIAN'}
        flux_ratios_fit_2 = magnification_function(**magnification_function_kwargs)

        if test_mode:

            if 'grid_rmax' in keywords_master.keys():
                grid_rmax = keywords_master['grid_rmax']
            else:
                grid_rmax = None

            print('flux ratios: ', flux_ratios_fit)
            print('flux ratios measured: ', np.array(keywords_master['fluxes']))
            cols = ['k', 'r', 'm', 'g']

            import matplotlib.pyplot as plt
            for i in range(0, 4):
                plt.scatter(data_to_fit.x[i], data_to_fit.y[i], color=cols[i], marker='+')
                plt.annotate(flux_ratios_fit[i], color=cols[i], xy=(data_to_fit.x[i], data_to_fit.y[i]))
                plt.annotate(np.array(keywords_master['fluxes'])[i], color=cols[i], xy=(data_to_fit.x[i], data_to_fit.y[i]-0.15))
            plt.show()

            ran = approx_theta_E(data_to_fit.x, data_to_fit.y)
            _x = _y = np.linspace(-1.5 * ran, 1.5 * ran, 200)
            xx, yy = np.meshgrid(_x, _y)
            shape0 = xx.shape
            kappa = lensModel_fit.kappa(xx.ravel(), yy.ravel(), kwargs_lens_fit).reshape(shape0)
            lensmodel_macro, kwargs_macro = lens_system.get_lensmodel(include_substructure=False)

            kappa_macro = lensmodel_macro.kappa(xx.ravel(), yy.ravel(), kwargs_macro).reshape(shape0)
            delta_kappa = kappa - kappa_macro
            plt.imshow(delta_kappa - np.mean(delta_kappa), vmin=-0.05, vmax=0.05, cmap='bwr', origin='lower',
                       extent=[-1.5 * ran, 1.5 * ran, -1.5 * ran, 1.5 * ran])
            plt.scatter(data_to_fit.x, data_to_fit.y, color='k')
            plt.show()
            lens_system.plot_images(data_to_fit.x, data_to_fit.y, source_samples['source_fwhm_pc'],
                             lensModel_fit,
                             kwargs_lens_fit,
                             grid_resolution=None,
                             grid_resolution_rescale=2,
                             grid_rmax=grid_rmax,
                             source_model=keywords_master['source_model'])
            a=input('continue')

        flux_ratios_fit = np.round(flux_ratios_fit, 5)
        kwargs_macro_ref = lens_system.macromodel.kwargs

        if readout_macro:
            comp1 = kwargs_e1e2_to_polar(lens_system.macromodel.components[0].kwargs[0])
            comp2 = kwargs_gamma1gamma2_to_polar(lens_system.macromodel.components[0].kwargs[1])
            kwargs_macro_new = {}
            for key in comp1.keys():
                kwargs_macro_new[key] = comp1[key]
            for key in comp2.keys():
                kwargs_macro_new[key] = comp2[key]
            kwargs_macro.append(kwargs_macro_new)

        if keywords_master['verbose']:
            print('flux_ratios_fit:', flux_ratios_fit)
            print('n remaining: ', keywords_master['Nsamples'] - (counter + 1))

        header = ''
        parameters = []
        for name in params_sampled.keys():
            header += name + ' '
            parameters.append(params_sampled[name])

        parameters = np.array(parameters)

        if save_best_realization:
            fluxes_measured = np.array(keywords_master['fluxes'])
            df = flux_ratios_fit[1:]/flux_ratios_fit[0] - fluxes_measured[1:]/fluxes_measured[0]
            new_statistic = np.sum(np.sqrt(df ** 2))
            if test_mode or keywords_master['verbose']:
                print('new statistic: ', new_statistic)
                print('current best statistic: ', current_best_statistic)
            if new_statistic < current_best_statistic:
                readout_best = True
                print('storing new realization...')
                current_best_statistic = new_statistic
                best_realization = SavedRealization(lensModel_fit, kwargs_lens_fit, flux_ratios_fit,
                                                    new_statistic, parameters)

        if fluxes_computed is None and parameters_sampled is None and fluxes_computed_2 is None:
            fluxes_computed = flux_ratios_fit
            fluxes_computed_2 = flux_ratios_fit_2
            parameters_sampled = parameters

        else:
            fluxes_computed = np.vstack((fluxes_computed, flux_ratios_fit))
            fluxes_computed_2 = np.vstack((fluxes_computed_2, flux_ratios_fit_2))
            parameters_sampled = np.vstack((parameters_sampled, parameters))

        if fluxes_computed is not None and (counter+1) % readout_steps == 0:
            t_end = time()
            t_ellapsed = t_end - t_start
            sampling_rate = fluxes_computed.shape[0] / t_ellapsed
            readout(readout_path, kwargs_macro, fluxes_computed, parameters_sampled,
                    header, write_header, write_mode, sampling_rate, readout_macro)
            readout(readout_path, kwargs_macro, fluxes_computed_2, parameters_sampled,
                    header, write_header, write_mode, sampling_rate, readout_macro, readout_flux_only=True,
                    flux_file_extension='_2')
            fluxes_computed, fluxes_computed_2, parameters_sampled = None, None, None
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
