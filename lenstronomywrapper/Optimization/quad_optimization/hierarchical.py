from lenstronomywrapper.Optimization.quad_optimization.brute import BruteOptimization
import numpy as np
from lenstronomywrapper.Optimization.quad_optimization.settings import *
from lenstronomywrapper.Utilities.lensing_util import interpolate_ray_paths

class HierarchicalOptimization(BruteOptimization):

    def __init__(self, lens_system, n_particles=None, simplex_n_iter=None, settings_class='default',
                 settings_kwargs={}):

        if settings_class == 'default':
            settings_class = HierarchicalSettingsDefault()
        elif settings_class == 'delta_function':
            settings_class = HierarchicalSettingsDeltaFunction(**settings_kwargs)
        else:
            settings_class = settings_class()

        if n_particles is None:
            n_particles = settings_class.n_particles
        if simplex_n_iter is None:
            n_iterations = settings_class.n_iterations

        self._n_particles = n_particles
        self._n_iterations = n_iterations

        self.settings = settings_class

        super(HierarchicalOptimization, self).__init__(lens_system,
                                                       log_mass_sheet_front=settings_class.log_mass_cut_global,
                                                       log_mass_sheet_back=settings_class.log_mass_cut_global)

    def optimize(self, data_to_fit, opt_routine='fixed_powerlaw_shear', constrain_params=None, verbose=False,
                 include_substructure=True):

        self._check_routine(opt_routine, constrain_params)

        realization = self.realization_initial

        if realization is not None:
            foreground_realization, background_realization = realization.split_at_z(self.lens_system.zlens)
        else:
            foreground_realization, background_realization = None, None

        foreground_rays, lens_model_raytracing, lens_model_full, foreground_realization_filtered, [source_x, source_y] = \
            self._fit_foreground(data_to_fit, foreground_realization, opt_routine, constrain_params, verbose)

        kwargs_lens_final, lens_model_raytracing, lens_model_full, info_array, source, realization_final = \
            self._fit_background(data_to_fit, foreground_realization_filtered, background_realization, foreground_rays,
                                 opt_routine, lens_model_raytracing, lens_model_full, source_x, source_y,
                                 constrain_params, verbose)

        return_kwargs = {'info_array': info_array,
                         'lens_model_raytracing': lens_model_raytracing,
                         'realization_initial': self.realization_initial,
                         'realization_final': realization_final}

        return self._return_results(source, kwargs_lens_final, lens_model_full, return_kwargs)

    def _fit_foreground(self, data_to_fit, realization_foreground, opt_routine, constrain_params=None, verbose=False):

        aperture_masses, globalmin_masses, window_sizes, scale, optimize_iteration, particle_swarm_reopt, \
        re_optimize_iteration = self.settings.foreground_settings

        N_foreground_halos_last = 0

        for run in range(0, self.settings.n_iterations_foreground):

            if run == 0:
                ray_x_interp, ray_y_interp = interpolate_ray_paths(data_to_fit.x, data_to_fit.y, self.lens_system,
                                                                   include_substructure=False)

            else:

                ray_x_interp, ray_y_interp = interpolate_ray_paths(data_to_fit.x, data_to_fit.y, self.lens_system,
                                                                   realization=realization_filtered)

            filter_kwargs = {'aperture_radius_front': window_sizes[run],
                             'aperture_radius_back': 0.,
                             'mass_allowed_in_apperture_front': aperture_masses[run],
                             'mass_allowed_in_apperture_back': 12,
                             'mass_allowed_global_front': globalmin_masses[run],
                             'mass_allowed_global_back': 10.,
                             'interpolated_x_angle': ray_x_interp,
                             'interpolated_y_angle': ray_y_interp,
                             'zmax': self.lens_system.zlens
                             }

            if run == 0:
                if realization_foreground is not None:
                    realization_filtered = realization_foreground.filter(**filter_kwargs)
                else:
                    realization_filtered = None

                if verbose: print('optimization ' + str(1))

            else:
                if realization_foreground is not None:
                    real = realization_foreground.filter(**filter_kwargs)
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
                    self.fit(data_to_fit, self._n_particles, opt_routine, constrain_params, self._n_iterations,
                              optimizer_kwargs, verbose, particle_swarm=particle_swarm_reopt[run],
                              re_optimize=re_optimize_iteration[run], tol_mag=None, realization=realization_filtered)

                N_foreground_halos_last = N_foreground_halos

                self.lens_system.update_kwargs_macro(kwargs_lens_final)
                self.lens_system.clear_static_lensmodel()

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

            if run == 0:
                ray_x_interp, ray_y_interp = interpolate_ray_paths(data_to_fit.x, data_to_fit.y, self.lens_system,
                                                                   realization=foreground_realization_filtered)
            else:
                ray_x_interp, ray_y_interp = interpolate_ray_paths(data_to_fit.x, data_to_fit.y, self.lens_system,
                                                                   realization=realization_filtered)

            filter_kwargs = {'aperture_radius_front': 10.,
                             'aperture_radius_back': window_sizes[run],
                             'mass_allowed_in_apperture_front': 10.,
                             'mass_allowed_in_apperture_back': aperture_masses[run],
                             'mass_allowed_global_front': 10.,
                             'mass_allowed_global_back': globalmin_masses[run],
                             'interpolated_x_angle': ray_x_interp,
                             'interpolated_y_angle': ray_y_interp,
                             'zmin': self.lens_system.zlens
                             }

            if run == 0:

                if foreground_realization_filtered is not None:
                    N_foreground_halos = foreground_realization_filtered.number_of_halos_before_redshift(self.lens_system.zlens)
                    real = realization_background.filter(**filter_kwargs)
                    realization_filtered = foreground_realization_filtered.join(real)

                else:
                    N_foreground_halos = 0
                    realization_filtered = None

                if verbose: print('optimization ' + str(1))

            else:

                if verbose: print('optimization ' + str(run + 1))

                if realization_filtered is not None:
                    real = realization_background.filter(**filter_kwargs)
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
                    self.fit(data_to_fit, self._n_particles, opt_routine, constrain_params,
                              self._n_iterations, optimizer_kwargs, verbose,
                              particle_swarm=particle_swarm_reopt[run],
                              re_optimize=re_optimize_iteration[run], tol_mag=None, realization=realization_filtered)

                reoptimized_realizations.append(realization_filtered)
                self.lens_system.update_kwargs_macro(kwargs_lens_final)

                N_background_halos_last = N_background_halos

            else:
                reoptimized_realizations.append(realization_filtered)

        info_array = (reoptimized_realizations, ray_x_interp, ray_y_interp)

        return kwargs_lens_final, lens_model_raytracing, lens_model_full, info_array, \
               [source_x, source_y], realization_filtered
