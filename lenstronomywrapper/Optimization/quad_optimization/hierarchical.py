from lenstronomywrapper.Optimization.quad_optimization.brute import BruteOptimization
import numpy as np
from lenstronomywrapper.Optimization.quad_optimization.settings import *

class HierarchicalOptimization(BruteOptimization):

    def __init__(self, lens_system, n_particles=None, simplex_n_iter=None, settings_class='default'):

        if settings_class == 'default':
            settings_class = HierarchicalSettingsDefault()
        elif settings_class == 'black_hole_lensing':
            settings_class = HierarchicalSettingsLowMass()
        else:
            raise Exception('settings class not recognized')

        if n_particles is None:
            n_particles = settings_class.n_particles
        if simplex_n_iter is None:
            n_iterations = settings_class.n_iterations

        self._n_particles = n_particles
        self._n_iterations = n_iterations

        self.settings = settings_class

        super(HierarchicalOptimization, self).__init__(lens_system)

    def optimize(self, data_to_fit, opt_routine='fixed_powerlaw_shear', constrain_params=None, verbose=False,
                 include_substructure=True):

        self._check_routine(opt_routine, constrain_params)

        realization = self.lens_system.realization

        if realization is not None:
            foreground_realization, background_realization = self._split_realization(data_to_fit, realization)
        else:
            foreground_realization, background_realization = None, None

        foreground_rays, lens_model_raytracing, lens_model_full, foreground_realization_filtered, [source_x, source_y] = \
            self._fit_foreground(data_to_fit, foreground_realization, opt_routine, constrain_params, verbose)

        kwargs_lens_final, lens_model_raytracing, lens_model_full, info_array, source = \
            self._fit_background(data_to_fit, foreground_realization_filtered, background_realization, foreground_rays,
                                 opt_routine, lens_model_raytracing, lens_model_full, source_x, source_y,
                                 constrain_params, verbose)

        return_kwargs = {'info_array': info_array, 'lens_model_raytracing': lens_model_raytracing}

        return self._return_results(source, kwargs_lens_final, lens_model_full, return_kwargs)

    def _split_realization(self, datatofit, realization):

        foreground = realization.filter(datatofit.x, datatofit.y, mindis_front=10,
                                        mindis_back=0, logmasscut_front=0,
                                        logabsolute_mass_cut_front=0,
                                        logmasscut_back=20,
                                        logabsolute_mass_cut_back=20)

        background = realization.filter(datatofit.x, datatofit.y, mindis_front=0,
                                        mindis_back=10, logmasscut_front=20,
                                        logabsolute_mass_cut_front=20,
                                        logmasscut_back=0,
                                        logabsolute_mass_cut_back=0)

        return foreground, background

    def _return_ray_path(self, x_opt, y_opt, lensModel, kwargs_lens):

        xpath, ypath = [], []

        for xi, yi in zip(x_opt, y_opt):
            _x, _y, redshifts, Tzlist = lensModel. \
                lens_model.ray_shooting_partial_steps(0, 0, xi,
                                                      yi, 0, self.lens_system.zsource, kwargs_lens)

            xpath.append(_x)
            ypath.append(_y)

        nplanes = len(xpath[0])

        x_path, y_path = [], []
        for ni in range(0, nplanes):
            arrx = np.array([xpath[0][ni], xpath[1][ni], xpath[2][ni], xpath[3][ni]])
            arry = np.array([ypath[0][ni], ypath[1][ni], ypath[2][ni], ypath[3][ni]])
            x_path.append(arrx)
            y_path.append(arry)

        return np.array(x_path), np.array(y_path), np.array(redshifts), np.array(Tzlist)

    def _fit_foreground(self, data_to_fit, realization_foreground, opt_routine, constrain_params=None, verbose=False):

        aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, \
        re_optimize_iteration = self.settings.foreground_settings

        N_foreground_halos_last = 0

        for run in range(0, self.settings.n_iterations_foreground):

            filter_kwargs = {'mindis_front': window_sizes[run], 'mindis_back': 100,
                             'source_x': self.lens_system.source_centroid_x, 'source_y': self.lens_system.source_centroid_y,
                             'logmasscut_front': globalmin_masses[run],
                             'logabsolute_mass_cut_front': aperture_masses[run], 'logmasscut_back': 12,
                             'logabsolute_mass_cut_back': 12, 'zmax': self.lens_system.zlens}

            if run == 0:
                if realization_foreground is not None:
                    realization_filtered = realization_foreground.filter(data_to_fit.x, data_to_fit.y, **filter_kwargs)
                else:
                    realization_filtered = None

                if verbose: print('optimization ' + str(1))

            else:
                if realization_foreground is not None:
                    real = realization_foreground.filter(data_to_fit.x, data_to_fit.y, **filter_kwargs)
                    realization_filtered = real.join(realization_filtered)
                if verbose: print('optimization ' + str(run + 1))

            if realization_foreground is not None:
                N_foreground_halos = realization_filtered.number_of_halos_before_redshift(self.lens_system.zlens)
            else:
                N_foreground_halos = 0

            self.lens_system.update_realization(realization_filtered)

            if verbose:
                print('aperture size: ', window_sizes[run])
                print('minimum mass in aperture: ', aperture_masses[run])
                print('minimum global mass: ', globalmin_masses[run])
                print('N foreground halos: ', N_foreground_halos)

            do_optimization = True

            if run > 0:
                if N_foreground_halos == 0:
                    do_optimization = False
                if N_foreground_halos == N_foreground_halos_last:
                    do_optimization = False
            if optimize_iteration[run] is False:
                do_optimization = False

            if do_optimization:

                optimizer_kwargs = {'re_optimize_scale': scale[run]}
                kwargs_lens_final, lens_model_raytracing, lens_model_full, foreground_rays, images, [source_x, source_y] = \
                    self._fit(data_to_fit, self._n_particles, opt_routine, constrain_params, self._n_iterations,
                              optimizer_kwargs, verbose, particle_swarm=particle_swarm_reopt[run],
                              re_optimize=re_optimize_iteration[run], tol_mag=None)

                N_foreground_halos_last = N_foreground_halos

            else:

                N_foreground_halos_last = N_foreground_halos

        return foreground_rays, lens_model_raytracing, lens_model_full, realization_filtered, [source_x, source_y]

    def _fit_background(self, data_to_fit, foreground_realization_filtered, realization_background, foreground_rays,
                        opt_routine, lens_model_raytracing, lens_model_full, source_x, source_y, constrain_params=None, verbose=False):

        aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, \
        re_optimize_iteration = self.settings.background_settings

        N_background_halos_last = 0

        backx, backy, background_Tzs, background_zs, reoptimized_realizations = [], [], [], [], []

        for run in range(0, self.settings.n_iterations_background):

            filter_kwargs = {'mindis_back': window_sizes[run], 'mindis_front': 0,
                             'source_x': source_x, 'source_y': source_y,
                             'logmasscut_back': globalmin_masses[run],
                             'logabsolute_mass_cut_back': aperture_masses[run], 'logmasscut_front': 12,
                             'logabsolute_mass_cut_front': 12, 'zmin': self.lens_system.zlens}

            if run == 0:

                if foreground_realization_filtered is not None:
                    N_foreground_halos = foreground_realization_filtered.number_of_halos_before_redshift(self.lens_system.zlens)
                    real = realization_background.filter(data_to_fit.x, data_to_fit.y, **filter_kwargs)
                    realization_filtered = foreground_realization_filtered.join(real)
                else:
                    N_foreground_halos = 0
                    realization_filtered = None

                if verbose: print('optimization ' + str(1))

            else:

                if verbose: print('optimization ' + str(run + 1))
                filter_kwargs.update({'ray_x': path_x, 'ray_y': path_y, 'path_redshifts': path_redshifts,
                                          'path_Tzlist': path_Tzlist})
                if realization_filtered is not None:
                    real = realization_background.filter(data_to_fit.x, data_to_fit.y, **filter_kwargs)
                    realization_filtered = realization_filtered.join(real)

            if realization_filtered is None:
                N_background_halos = 0
            else:
                N_background_halos = realization_filtered.number_of_halos_after_redshift(self.lens_system.zlens)

            self.lens_system.update_realization(realization_filtered)

            if verbose and realization_filtered is not None:
                ntotal_halos = realization_filtered.number_of_halos_after_redshift(0)
                assert ntotal_halos == N_foreground_halos + N_background_halos

            if verbose:
                print('nhalos: ', N_background_halos+N_foreground_halos)
                print('aperture size: ', window_sizes[run])
                print('minimum mass in aperture: ', aperture_masses[run])
                print('minimum global mass: ', globalmin_masses[run])
                print('N foreground halos: ', N_foreground_halos)
                print('N background halos: ', N_background_halos)

            do_optimization = True
            if run > 0:
                if N_background_halos == 0:
                    do_optimization = False
                if N_background_halos == N_background_halos_last:
                    do_optimization = False
            if optimize_iteration[run] is False:
                do_optimization = False

            if do_optimization:

                optimizer_kwargs = {'save_background_path': True,
                                  're_optimize_scale': scale[run],
                                  'precomputed_rays': foreground_rays}

                kwargs_lens_final, lens_model_raytracing, lens_model_full, foreground_rays, images, [source_x, source_y] = \
                    self._fit(data_to_fit, self._n_particles, opt_routine, constrain_params,
                              self._n_iterations, optimizer_kwargs, verbose,
                              particle_swarm=particle_swarm_reopt[run], re_optimize=re_optimize_iteration[run], tol_mag=None)

                path_x, path_y, path_redshifts, path_Tzlist = self._return_ray_path(data_to_fit.x, data_to_fit.y, lens_model_full,
                                                        kwargs_lens_final)

                backx.append(path_x)
                backy.append(path_y)
                background_Tzs.append(path_Tzlist)
                background_zs.append(path_redshifts)
                reoptimized_realizations.append(realization_filtered)
                N_background_halos_last = N_background_halos

            else:
                reoptimized_realizations.append(realization_filtered)

        info_array = (backx, backy, background_Tzs, background_zs, reoptimized_realizations)

        return kwargs_lens_final, lens_model_raytracing, lens_model_full, info_array, [source_x, source_y]

